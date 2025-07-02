
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
print(df.head())
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label_num']
cv = CountVectorizer()
X_cv = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
sample_msg = ["Congratulations! You've won a free ticket to Dubai!"]
sample_vector = cv.transform(sample_msg)
print("Prediction:", model.predict(sample_vector))  