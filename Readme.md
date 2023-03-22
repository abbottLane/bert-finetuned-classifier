# Classification Model Results and Discussion

## How to run model:
I've used a makefile to manage a python virtual environment and run the model training and evaluation.
Simply type:
`make run` 
from the project directory. This will install all the required dependencies, train the model, and output the results on the test set.

## Model

Because there was not a lot of data, I decided to go the transfer learning route. I fine-tuned a single linear classification layer with dropout on the`bert-base-uncased` model from the Huggingface repository. I used an Adam optimizer with decoupled weight decay regularization (https://arxiv.org/abs/1711.05101) with the following hyperparameters:

```
LEARNING_RATE = 1e-3 # for AdamW optimizer
WEIGHT_DECAY = 1e-2 # for AdamW optimizer
EPSILON = 1e-8 # for AdamW optimizer
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
EPOCHS =50
TRAIN_SPLIT = .80
EXCLUDE_LABELS = {}
```
The model achieved these results on the test set:

```
               precision  recall  f1-score  support
business       0.547619    0.92  0.686567    50.00
scitech        0.793651    1.00  0.884956    50.00
sports         0.980392    1.00  0.990099    50.00
world          1.000000    0.04  0.076923    50.00
accuracy       0.740000    0.74  0.740000     0.74
macro avg      0.830415    0.74  0.659636   200.00
weighted avg   0.830415    0.74  0.659636   200.00
```

## Analysis Data

The main training script produces some figures and data in the `reports/` folder. Included here are:
- `{data_split}_{body|subject}_wordcount_{label}.csv`
    - This file is a CSV of word counts sorted descending by count. This helped me to examine lexical patterns by dataset (test or training), text content type (body or subject), and label (business, scitech, sports, or world)
-  `{data_split}_{body|subject}_wordcount_{label}.png`
    - A matplot lib figure of the above data
- `{data_split}_email_class_distribution.csv`
    - This file shows the distribution of email labels, per data split (training, test)
- `{data_split}_email_class_distribution.png`
    - A matplotlib figure of the class distribution
- `train_loss.png`
    - A figure showing the reported run's training loss curve

## Learnings from the analysis data

1. There is a big class imbalance. On another iteration of this project I would first try to up/downsample to get a distribution that matches our expectations of the real world.
2. The word distributions show that only scitech and business are following regular lexical and syntactic patterns of english. In English, we expect to see something zipfian with functional words like determiners and conjunctions at the top. In this dataset, this is only true for scitech and business, which, on the surface, seems to point to corrupt data.
3. However, while the sports class appears to be lexically scrambled, it is still zipfian. I inferred from this that its possible that the noise is not random, but ciphered by a regular vocabulary shift, and therefore even if it is not human readable it might still contain learnable underlying structure. The abilitiy of the model to learn from trining and generalize to the test set seems to confirm this theory.
4. Even if the scrambling is regular and retains syntactic structure, the vocabulary may still overlap with the non-scrambled classes. Because we are using a pretrained model as our base, there is a certain prior about word meanings and relationships. During fine-tuning, vocabulary which overlaps between scrambled and non-scrambled classes will cause noisy signal. This overlap, and the nature of the scramble might be worth digging into.          
3. The class imbalance issue is worse if we assume the test set reflects the expected "in-the-wild" distributions. This suggests a severe mismatch of expectation with respect to our data collection process vs. real world conditions. A test set should reflect real-world distributions, and validation data should be drawn from the same distribution so that we can have confidence in training that we are capturing signal representative of our use case.

There is plenty more to investigate and update on this project to improve performance, or gather evidence to eliminate some data.
Some ideas moving forward:

- implementing a resample of the classes to get a uniform distribution
- systematic ablation study to see how different factors affect performance, and how much
- further review of individual class data quality, led by the data points already collected.
- More extensive data cleaning and preprocessing
- investigating whether the test data represents real-world distribution of labels and features.