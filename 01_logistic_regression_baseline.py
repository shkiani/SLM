'''
 1. Install important dependencies
 pip install -r requirements.txt

 2. Import important dependencies
'''

from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report


'''3. Preprocess data'''
# 3.1 Load the dataset
train_raw = load_dataset("ag_news", split="train")
test_raw = load_dataset("ag_news", split="test")

# 3.2 Convert to pandas DataFrame
train = pd.DataFrame(train_raw)
test = pd.DataFrame(test_raw)

# Now you can use pandas methods for visualization
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

# 3.4 Vectorizes pandas dataframe into TF-IDF feature vectors
vectorizer = TfidfVectorizer()
vec_train_features = vectorizer.fit_transform(train_features)
vec_validation_features = vectorizer.transform(validation_features)
vec_test_features = vectorizer.transform(test['text'])

# Visualize feature vectors sample data and shapes: train (108000,62584), validation (12000,62584), test (7600, 62584)
print(vec_train_features[0])
print(vec_validation_features[0])
print(vec_test_features[0])
print(vec_train_features.shape)
print(vec_validation_features.shape)
print(vec_test_features.shape)

'''4. Train the model using training data'''
model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='saga')
model.fit(vec_train_features, train_labels)

'''5. Evaluate the model'''

# 5.1 predict on validation set
predicted_validation_labels = model.predict(vec_validation_features)
print('Model validation prediction', validation_labels[0:5])
print('Model validation real value', predicted_validation_labels[0:5])

probability_validation_labels = model.predict_proba(vec_validation_features)
print('Model validation prediction, probability', probability_validation_labels[0:5])

cm = confusion_matrix(validation_labels, predicted_validation_labels)
print("Confusion Matrix:\n", cm)

score = model.score(vec_validation_features, validation_labels)
print('Model Accuracy Score', score)

macro_precision = precision_score(validation_labels, predicted_validation_labels, average='macro') #calculate precision for all classes individually and then average them
macro_recall = recall_score(validation_labels, predicted_validation_labels, average='macro')
micro_precision = precision_score(validation_labels, predicted_validation_labels, average='micro') #calculate class wise true positive and false positive and then use that to calculate overall precision
micro_recall = recall_score(validation_labels, predicted_validation_labels, average='micro')
print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Micro Precision:", micro_precision)
print("Micro Recall:", micro_recall)

# 5.2 Predict on test set
predicted_test_labels = model.predict(vec_test_features)
probability_test_labels = model.predict_proba(vec_test_features)

# 5.3 Evaluate
test_score = model.score(vec_test_features, test['label'])
print("Test Accuracy Score:", test_score)

print("Classification Report:\n", classification_report(test['label'], predicted_test_labels))
