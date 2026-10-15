import pandas as pd
import numpy as np
import re
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

train_data = pd.read_csv("train.csv", sep=';', engine='python')
valid_classes = [
    'Cardiovascular-Pulmonary', 'Dermatology', 'Gastroenterology',
    'General Medicine', 'Neurology', 'Neurosurgery',
    'Obstetrics-Gynecology', 'Ophthalmology', 'Orthopedic',
    'Psychiatry-Psychology', 'Radiology', 'Surgery'
]

train_data['medical_specialty'] = train_data['medical_specialty'].astype(str).str.strip()
train_data = train_data[train_data['medical_specialty'].isin(valid_classes)].reset_index(drop=True)

with open("test_no_labels.csv", 'r', encoding='utf-8', errors='replace') as f:
    header_check = f.readline().strip().lower()

if header_check.startswith('description;'):
    test_data = pd.read_csv("test_no_labels.csv", sep=';', engine='python')
else:
    test_data = pd.read_csv("test_no_labels.csv", sep=';', header=None, names=['description', 'sample_name', 'transcription', 'keywords'], engine='python')

def clean_text(text):
    if type(text) != str:
        return ""
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s\-\/]', ' ', t)
    return " ".join(t.split())

for df in [train_data, test_data]:
    df['description'] = df['description'].fillna('').apply(clean_text)
    df['sample_name'] = df['sample_name'].fillna('').apply(clean_text)
    df['keywords'] = df['keywords'].fillna('').apply(clean_text)
    df['transcription'] = df['transcription'].fillna('').apply(clean_text)
    df['all_text'] = df['sample_name'] + " " + df['description'] + " " + df['keywords'] + " " + df['transcription']

def drop_words(text, prob=0.15):
    words = text.split()
    if len(words) < 5:
        return text
    kept = [w for w in words if random.random() > prob]
    return " ".join(kept)

def augment_data(df, min_count=150):
    counts = df['medical_specialty'].value_counts()
    new_data = []
    for cat, count in counts.items():
        if count < min_count:
            sub = df[df['medical_specialty'] == cat]
            missing = min_count - count
            for _ in range(missing):
                row = sub.sample(1).iloc[0].copy()
                row['transcription'] = drop_words(row['transcription'])
                row['description'] = drop_words(row['description'])
                row['all_text'] = row['sample_name'] + " " + row['description'] + " " + row['keywords'] + " " + row['transcription']
                new_data.append(row)
    if len(new_data) > 0:
        return pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    return df

print("Starting cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true = train_data['medical_specialty'].values
labels = np.unique(y_true)
oof_preds = np.zeros(len(train_data), dtype=object)

for train_idx, val_idx in skf.split(train_data, y_true):
    train_fold = train_data.iloc[train_idx]
    val_fold = train_data.iloc[val_idx]
    
    train_fold_aug = augment_data(train_fold, 150)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_fold_aug['all_text'])
    y_train = train_fold_aug['medical_specialty'].values
    
    X_val = vectorizer.transform(val_fold['all_text'])
    
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    
    oof_preds[val_idx] = model.predict(X_val)

print("\nCV Results:")
print("Accuracy:", accuracy_score(y_true, oof_preds))
f1s = f1_score(y_true, oof_preds, average=None, labels=labels)
print("Min F1:", np.min(f1s))
print("\n", classification_report(y_true, oof_preds, digits=4))

cm = confusion_matrix(y_true, oof_preds, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=200)
plt.close()

print("\nTraining final model...")
final_train = augment_data(train_data, 150)
final_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
X_final = final_vec.fit_transform(final_train['all_text'])
y_final = final_train['medical_specialty'].values

final_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=300, random_state=42)
final_model.fit(X_final, y_final)

X_test = final_vec.transform(test_data['all_text'])
test_predictions = final_model.predict(X_test)

with open("results.txt", "w", encoding="utf-8") as file:
    for p in test_predictions:
        file.write(p + "\n")

print("File results.txt generated successfully.")