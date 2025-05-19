# Small Language Models

## Introduction
 
This is a modular series for learning **large language models (LLMs)** and starts with **small language models (SLMs)** that can run locally without API dependencies.

We walk through:

1. Traditional classification model `01_logistic_regression_baseline.py`
2. Finetuning a small transformer `02_finetune_slm.py`


## Overview 

| **File** | **Model** | **Task** | **Dataset** | **Framework** |
|----------|-------|-------|-------|-------|
| `01_logistic_regression_baseline.py` | `LogisticRegression` | Text Classification | AG News (4 categories, 127,600 rows)| Scikit-learn (for classical ML) |
| `02_finetune_slm.py` | `bert-base-uncased` | Text Classification | AG News (4 categories, 127,600 rows)| Hugging Face Transformers |


## Discussion

### Preprocessing 

1. In `01_logistic_regression_baseline.py` the AG News dataset is converted to a DataFrame, split into training/validation sets, and transformed into TF-IDF features, capturing term frequency and inverse document frequency without preserving contextual structure, suitable for linear models.

2. In `02_finetune_slm.py` the AG News dataset is converted to a DataFrame, split into training/validation sets, and tokenized using a BERT-compatible tokenizer that preserves word contex, then wrapped into Dataset objects.

### Model Architecture:
1. In `01_logistic_regression_baseline.py`, a multinomial logistic regression model is trained using the saga solver with epochs up to 500 iterations (auto-stop on convergence), and implicit, adaptive learning rate inside solver.
 
2. In `02_finetune_slm.py`, a pre-trained BERT model, adapted for multi-class text classification (AutoModelForSequenceClassification), plus a new classification head on top (a fully connected layer with output size matching the number of classes) is fine-tuned using Hugging Face's Trainer. Training was done over 3 epochs with a learning rate of 2e-5.

### Evaluation:
Performance evaluation is done using metrics like accuracy and classification report on tokenized test data, confusion matrix, precision, recall, accuracy, and classification report on TF-IDF feature vectors for validation and test sets.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
