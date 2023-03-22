import pickle
import string
import torch
import numpy as np
from sklearn import metrics
from torch import nn
from torch.utils.data.dataset import random_split
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel
import matplotlib.pyplot as plt
from BertDataset import get_dataloader
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-3 # for AdamW optimizer
WEIGHT_DECAY = 1e-2 # for AdamW optimizer
EPSILON = 1e-8 # for AdamW optimizer
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
EPOCHS =50
TRAIN_SPLIT = .80
EXCLUDE_LABELS = {}

def main(train_path: str, test_path: str) -> None:
    ### Step #0: Load data
    train_data = load_data(train_path)

    ### Step #1: Analyse data
    analyse_data(train_data, split="train")

    ### Step #2: Clean and prepare data
    train_data = data_clean(train_data, exclude_labels=EXCLUDE_LABELS)
    train_ds, val_ds, labels = data_prepare(train_data, split_percent = TRAIN_SPLIT)

    ### Step #3: Extract features
    train_iter, val_iter = extract_features(train_ds, val_ds, labels)
    
    ### Step #4: Train model
    model = EmailClassifier(num_classes=len(labels))
        
    criterion = nn.CrossEntropyLoss()
    optimizer= AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON, weight_decay=WEIGHT_DECAY)
    
    loss_data=train_model(EPOCHS, train_iter, val_iter, model, criterion, optimizer)
    plot_loss(loss_data, phase="train")

    ### Step #5: Stand-alone Test data & Compute metrics
    test_data = load_data(test_path)
    analyse_data(test_data, split="test")
    test_data = data_clean(test_data, exclude_labels=EXCLUDE_LABELS)
    test_data, _, labels = data_prepare(test_data, split_percent = 1)
    # Extract features is just a wrapper around the dataloader, so we can use it directly here
    test_data = get_dataloader(test_data, labels, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    compute_metrics(model, test_data, phase='test', label_names=test_data.dataset.label_map.values())
    return 0


class EmailClassifier(nn.Module):
    """A BERT-based email classifier
    
    Given that there is not a lot of training data, we will use a pre-trained BERT model and fine-tune it for classification. 
    This simple model loads the pre-trained BERT model and adds a dropout layer and a linear layer on top of it.
    During training we will freeze the BERT model and only train the linear layer.

    Args:
        num_classes (int): The number of classes to classify emails into. Defaults to NUM_CLASSES.
    """
    def __init__(self, num_classes):
        super(EmailClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # We do not want to retrain our pretrained layers except the last linear layer
        for param in self.bert.parameters():
            param.requires_grad = False

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
    write_wordcount_and_distribution([(x['Subject'], x['Label']) for x in data], split, str_type='subject')
    write_wordcount_and_distribution([(x['Body'], x['Label']) for x in data], split, str_type='body')
    write_label_distribution(data, split)

# Step #2: Clean data
def data_clean(data: list, exclude_labels: set) -> list:
    """A data cleaning routine. Removes punctuation and unwanted whitespace from the data. """
    clean_data = []
    for curr_data in data:
        if curr_data['Label'] not in EXCLUDE_LABELS:
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
def data_prepare(data: list, split_percent: int, resample: bool = False) -> list:
    """A data preparation routine. If this is training, splits data into train and validation sets.
    If this is testing, split_percent should be set to 1 so no splitting occurs.
    """
    # make a label map
    labels = sorted(list(set([x['Label'] for x in data])))
    label_map= dict(zip(labels, range(len(labels))))

    # reshape data into string label pairs
    dataset = [(label_map[x['Label']], x['text']) for x in data]

    if resample: 
        # resample data to balance classes
        dataset = resample_data(dataset)
        
    # split data into train and validation sets if this is training
    num_train = split_percent * len(dataset)
    split_train_, split_valid_ = random_split(dataset, [int(num_train), len(dataset) - int(num_train)])
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

        running_loss = []
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

            running_loss.append(float(loss.item()))

            # Print training progress every 22 steps/ 
            num_steps = 22
            if step % num_steps == 0:
                # compute the average running loss
                arl = (sum(running_loss) / len(running_loss))
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {arl:.4f}')
                loss_data.append(("epoch: " + str(epoch + 1) + " step: " + str(step), arl))
                running_loss = []
        
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
    plt.gcf().subplots_adjust(bottom=0.2) # make room for x-axis labels
    plt.xticks(np.arange(0, len(x), 5)) # spread out x axis ticks so you can read labels
    plt.ylabel("Loss")
    plt.title(f"{phase} loss")
    plt.legend()
    plt.savefig(f"./reports/{phase}_loss.png")
    plt.clf()

def write_label_distribution(data, split):
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
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'./reports/{split}_email_class_distribution.png')
    plt.clf()

def write_wordcount_and_distribution(data, split, str_type):
    # for every label, collect words and counts
    tokens_by_label = {label: {} for label in set([x[1] for x in data])}
    for x in data:
        tokens = x[0].split()
        for token in tokens:
            if token in tokens_by_label[x[1]]:
                tokens_by_label[x[1]][token] += 1
            else:
                tokens_by_label[x[1]][token] = 1
    for label, tokens in tokens_by_label.items():
        # Write word counts to a csv file
        with open(f'./reports/{split}_{str_type}_wordcount_{label}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'count', 'percentage'])
            total = sum(tokens.values())
            # sort by count
            sorted_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
            for word, count in sorted_tokens:
                writer.writerow([word, count, count / total])
        
        # make a plot of the top 20 words
        sorted_tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
        top_20 = sorted_tokens[:20]
        x, y = zip(*top_20)
        plt.bar(x, y)
        plt.xlabel("Word")
        plt.ylabel("Count")
        plt.gcf().autofmt_xdate()
        plt.title(f"Top 20 words in {split} set for label {label}")
        plt.savefig(f"./reports/{split}_{str_type}_wordcount_{label}.png")
        plt.clf()

def resample_data(data, split):
    pass

if __name__ == "__main__":
    train_path = "./data/agnews_combined_train.pkl"
    test_path = "./data/agnews_combined_test.pkl"
    main(train_path, test_path)