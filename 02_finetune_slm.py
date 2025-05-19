'''
 1. Install important dependencies
 pip install -r requirements.txt

 2. Import important dependencies
'''
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

'''3. Preprocess data'''
# 3.1 Load the dataset
train_raw = load_dataset("ag_news", split="train")
test_raw = load_dataset("ag_news", split="test")

# 3.2 Convert to pandas DataFrame
train = pd.DataFrame(train_raw)
test = pd.DataFrame(test_raw)

# Now you can use pandas methods:
print(train.head())        # Show first 5 rows
print(train.info())        # Summary info about columns and data types
print(train.describe())    # Statistical summary of numerical columns (if any)
print(train.columns)       # List of column names

# 3.3 Split into train and validation set
train_features, validation_features, train_labels, validation_labels = train_test_split(
    train['text'], train['label'], test_size=0.1, random_state=42)

# Visualize sample data and shape of each => validation is (12000,) and train is (108000,)
print(train_features.iloc[0])
print(validation_features.iloc[0])
print(train_features.shape)
print(validation_features.shape)


# 3.4 Tokenize pandas dataframe into model-compatible tokenizer
model_name = "google-bert/bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

enc_train_features = tokenizer(list(train_features), truncation=True, padding=True)
enc_validation_features = tokenizer(list(validation_features), truncation=True, padding=True)
enc_test_features = tokenizer(list(test['text']), truncation=True, padding=True)

print(enc_train_features.keys())
print("Input IDs:", enc_train_features["input_ids"][0])
print("Attention Mask:", enc_train_features["attention_mask"][0])

# 3.5 Wrap the tokenized data into Dataset objects.
class LLMClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = LLMClassificationDataset(enc_train_features, list(train_labels))
validation_dataset = LLMClassificationDataset(enc_validation_features, list(validation_labels))
test_dataset = LLMClassificationDataset(enc_test_features, list(test['label']))

'''4. Train the model using training data'''
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    no_cuda = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

'''5. Evaluate the model'''
trainer.evaluate()

preds_output = trainer.predict(enc_test_features)
preds = np.argmax(preds_output.predictions, axis=1)


true_test_labels = test['label']
print("SLM Test Accuracy:", accuracy_score(true_test_labels, preds))
print("SLM Classification Report:\n", classification_report(true_test_labels, preds))