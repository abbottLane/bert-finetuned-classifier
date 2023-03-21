import pickle
import string
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data.dataset import random_split
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel
import matplotlib.pyplot as plt
from BertDataset import get_dataloader
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 4
TRAIN_SPLIT = .80

class EmailClassifier(nn.Module):
    """A BERT-based email classifier
    
    Given that there is not a lot of training data, we will use a pre-trained BERT model and fine-tune it for classification. 
    This simple model loads the pre-trained BERT model and adds a dropout layer and a linear layer on top of it.
    During training we will freeze the BERT model and only train the linear layer.

    Args:
        num_classes (int): The number of classes to classify emails into. Defaults to NUM_CLASSES.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmailClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Step #0: Load data
def load_data(path: str) -> list:
    """Load Pickle files"""
    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

# Step #1: Analyse data
def analyse_data(data: list, split: str) -> None:
    """Analyse data files"""
    # Check the number of emails in each class and write a csv file with the headers [label, count, percentage]
    labels = [x['Label'] for x in data]
    with open(f'./reports/{split}_email_class_distribution.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'count', 'percentage'])
        for label in sorted(list(set(labels))):
            count = labels.count(label)
            writer.writerow([label, count, count / len(labels)])

    # plot the number of emails in each class and write to a file
    pd.Series(labels).value_counts().plot(kind='bar')
    plt.subplots_adjust(bottom=0.6)
    plt.savefig(f'./reports/{split}_email_class_distribution.png')
    plt.clf()

    # count occurences of each word in the emails and plot a new histogram
    words_and_counts = {}
    for x in data:
        for word in x['Body'].split():
            if word in words_and_counts:
                words_and_counts[word] += 1
            else:
                words_and_counts[word] = 1
        for word in x['Subject'].split():
            if word in words_and_counts:
                words_and_counts[word] += 1
            else:
                words_and_counts[word] = 1
    pd.Series(words_and_counts).value_counts().sort_values(ascending=True).plot(kind='hist')
    plt.savefig(f'./reports/{split}_email_word_distribution.png')
    plt.clf()

    # write word count to a csv file
    with open(f'./reports/{split}_email_word_distribution.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['word_count'])
        for x in data:
            writer.writerow([len(x['Body'].split())])


# Step #2: Clean data
def data_clean(data: list) -> list:
    """A data cleaning routine. Removes punctuation and unwanted whitespace from the data. """
    clean_data = []
    for curr_data in data:
        # Remove punctuation
        curr_data["Subject"] = curr_data["Subject"].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        curr_data["Body"] = curr_data["Body"].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        # Remove whitespace (\t)
        curr_data["Body"] = curr_data["Body"].replace("\t", " ")
        curr_data["Subject"] = curr_data["Subject"].replace("\t", " ")
        curr_data['text'] = curr_data['Subject'] + " " + curr_data['Body']
        clean_data.append(curr_data)        
    return clean_data

# Step #2: Prepare data
def data_prepare(train_data: list, train_percent: int) -> list:
    """A data preparation routine. Splits data into train and validation sets.
    """
    # make a label map
    labels = sorted(list(set([x['Label'] for x in train_data])))
    label_map= dict(zip(labels, range(len(labels))))

    # reshape data into string label pairs
    train_dataset = [(label_map[x['Label']], x['text']) for x in train_data]
    num_train = train_percent * len(train_dataset)
    split_train_, split_valid_ = random_split(train_dataset, [int(num_train), len(train_dataset) - int(num_train)])
    train_data_split = data_by_indices(split_train_)
    valid_data_split = data_by_indices(split_valid_)
    return train_data_split, valid_data_split, label_map

# Step #3: Extract features
def extract_features(X_train: list, X_valid: list, labels:dict()) -> list:
    """Extract features from the data"""
    train_dataloader= get_dataloader(X_train, labels, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    val_dataloader= get_dataloader(X_valid, labels, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    return train_dataloader, val_dataloader

# Step #4: Train model
def train_model(epochs,train_dataloader,val_dataloader,model,criterion,optimizer) -> EmailClassifier: #: tt.data.BucketIterator
    """Create a training loop"""
    #Define the scheduler
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1 * total_steps)  # Warm up to 10% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    loss_data = [] # collect loss data for plotting
    for epoch in range(epochs):
        model.train()
        model.to(device)

        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Print training progress every 10 steps
            if step % 10 == 9:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}')
                loss_data.append(("epoch: " + str(epoch + 1) + " step: " + str(step), running_loss / 100))
                running_loss = 0.0
        
        # Calculate accuracy on validation set
        compute_metrics(model, val_dataloader, phase='validation', label_names=val_dataloader.dataset.label_map.values())

    print('Training complete.')
    return loss_data


# Step #5: Stand-alone Test data & Compute metrics
def compute_metrics(model: EmailClassifier, dataloader: list, phase: str, label_names: list) -> None:
    """Compute metrics for the model given a dataset"""
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predicted = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        
        # calculate P, R, F1, and Acc by label
        report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=label_names)
        df = pd.DataFrame(report).transpose()
        print(f'current performance on {phase} set: \n {df}')

    
def data_by_indices(dataset):
    """Returns a list of data rows given a dataset and a list of indices"""
    return [dataset.dataset[i] for i in dataset.indices]



def main(train_path: str, test_path: str) -> None:
    ### Perform the following steps and complete the code

    ### Step #0: Load data
    train_data = load_data(train_path)

    ### Step #1: Analyse data
    analyse_data(train_data, split="train")

    ### Step #2: Clean and prepare data
    # fields, TEXTFIELD, LABELFIELD = data_fields()
    train_data = data_clean(train_data)
    train_ds, val_ds, labels = data_prepare(train_data, train_percent = TRAIN_SPLIT)

    ### Step #3: Extract features
    train_iter, val_iter = extract_features(train_ds, val_ds, labels)
    
    ### Step #4: Train model
    model = EmailClassifier()
    
    # We do not want to retrain our pretrained layers except the last linear layer
    for param in model.bert.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(),lr= 0.0001)
    
    loss_data=train_model(EPOCHS, train_iter, val_iter, model, criterion, optimizer)
    plot_loss(loss_data, phase="train")

    ### Step #5: Stand-alone Test data & Compute metrics
    test_data = load_data(test_path)
    analyse_data(test_data, split="test")
    test_data = data_clean(test_data)
    test_data = get_dataloader(test_data, labels, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    compute_metrics(model, test_data, phase='test', label_names=test_data.dataset.label_map.values())
    return 0

def plot_loss(loss_data, phase):
    """Plot the training loss data
    Args:
        loss_data (list): list of tuples containing epoch and loss
        phase (str): train or validation
        
    Returns:
        None
    """
    x, y = zip(*loss_data)
    plt.plot(x, y, label=phase)
    plt.xlabel("Epoch")
    plt.gcf().autofmt_xdate() # rotate x-axis labels
    plt.ylabel("Loss")
    plt.title(f"{phase} loss")
    plt.legend()
    plt.savefig(f"./reports/{phase}_loss.png")
    plt.clf()

if __name__ == "__main__":
    train_path = "./data/agnews_combined_train.pkl"
    test_path = "./data/agnews_combined_train.pkl"
    main(train_path, test_path)