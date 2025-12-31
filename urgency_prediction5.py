import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Data Preparation for Urgency
print("Step 1: Data Preparation for Urgency")
df = pd.read_csv('labeled_emails.csv')
print(f"Loaded {len(df)} labeled emails.")

df['cleaned_text'] = df['cleaned_text'].fillna('')
df = df[df['cleaned_text'] != '']
print(f"After cleaning: {len(df)} emails.")

X = df['cleaned_text']
y = df['urgency']  #urgency levels ('low', 'medium', 'high')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("Data prepared for urgency classification.")

# Step 2: Keyword Based Urgency Detection
print("\nStep 2: Keyword Based Urgency Detection")
def keyword_urgency_detection(text):
    keywords_high = ['urgent', 'asap', 'deadline', 'emergency', 'immediate', 'critical']
    keywords_medium = ['soon', 'important', 'priority', 'quick']
    text_lower = text.lower()
    if any(word in text_lower for word in keywords_high):
        return 'high'
    elif any(word in text_lower for word in keywords_medium):
        return 'medium'
    else:
        return 'low'

# Apply to test data for validation
df_test_sample = pd.DataFrame({'cleaned_text': X_test, 'actual_urgency': y_test})
df_test_sample['keyword_urgency'] = df_test_sample['cleaned_text'].apply(keyword_urgency_detection)
print("Keyword Detection Sample:")
print(df_test_sample[['cleaned_text', 'keyword_urgency']].head())

# Step 3: Train Baseline Classifiers for Urgency
print("\nStep 3: Training Baseline Classifiers for Urgency")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(classification_report(y_test, y_pred_nb, zero_division=0))

# Step 4: Combine ML + Keyword Based Detection
print("\nStep 4: Combining ML and Keyword Based Detection")
def combined_urgency_prediction(ml_prediction, text):
    keyword_pred = keyword_urgency_detection(text)
    if keyword_pred == 'high':
        return 'high'
    else:
        return ml_prediction

# Combine for Logistic Regression
y_pred_lr_combined = [combined_urgency_prediction(pred, text) for pred, text in zip(y_pred_lr, X_test)]
print("Combined Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_combined):.4f}")
print(classification_report(y_test, y_pred_lr_combined, zero_division=0))

# Combine for Naive Bayes
y_pred_nb_combined = [combined_urgency_prediction(pred, text) for pred, text in zip(y_pred_nb, X_test)]
print("Combined Naive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb_combined):.4f}")
print(classification_report(y_test, y_pred_nb_combined, zero_division=0))

# Step 5: Fine Tune DistilBERT for Urgency
print("\nStep 5: Fine Tuning DistilBERT for Urgency")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
num_labels = len(label_encoder.classes_)
print(f"Urgency Classes: {label_encoder.classes_}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')  # Reduced for speed

train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())
test_encodings = tokenize_function(X_test.tolist())

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, y_train_encoded)
val_dataset = EmailDataset(val_encodings, y_val_encoded)
test_dataset = EmailDataset(test_encodings, y_test_encoded)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Predict
predictions = trainer.predict(test_dataset)
y_pred_probs = predictions.predictions
y_pred = y_pred_probs.argmax(axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred)

y_pred_bert_combined = [combined_urgency_prediction(pred, text) for pred, text in zip(y_pred_labels, X_test)]

print("Combined DistilBERT Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bert_combined):.4f}")
print(classification_report(y_test, y_pred_bert_combined, zero_division=0))

# Step 6: Evaluate Classification Accuracy with Confusion Matrices and F1 Scores
print("\nStep 6: Evaluation with Confusion Matrices and F1 Scores")

cm_lr = confusion_matrix(y_test, y_pred_lr_combined)
plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Combined Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"Combined LR F1 Score (Macro): {f1_score(y_test, y_pred_lr_combined, average='macro'):.4f}")
print(f"Combined LR F1 Score (Weighted): {f1_score(y_test, y_pred_lr_combined, average='weighted'):.4f}")

cm_bert = confusion_matrix(y_test, y_pred_bert_combined)
plt.figure(figsize=(8,6))
sns.heatmap(cm_bert, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Combined DistilBERT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"Combined BERT F1 Score (Macro): {f1_score(y_test, y_pred_bert_combined, average='macro'):.4f}")
print(f"Combined BERT F1 Score (Weighted): {f1_score(y_test, y_pred_bert_combined, average='weighted'):.4f}")

print("Executed Successfully")