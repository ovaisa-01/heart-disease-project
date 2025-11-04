# heart_disease.py
# Beginner Machine Learning Project — Heart Disease Prediction
# Author: Ovaisa

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
data = pd.read_csv('heart.csv')

print("✅ Dataset loaded successfully!")
print("Shape of dataset:", data.shape)
print("\nPreview of data:\n", data.head())

# 2. Check for missing values
print("\nMissing values in each column:\n", data.isnull().sum())

# 3. Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# 4. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 5. Feature scaling (normalize data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Display a Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n✅ Done! Heart disease prediction model completed successfully.")
