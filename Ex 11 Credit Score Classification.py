import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Income': [50000, 60000, 30000, 80000, 20000, 90000, 40000, 70000],
    'Age': [25, 45, 35, 50, 23, 55, 40, 48],
    'LoanAmount': [200000, 150000, 300000, 100000, 350000, 90000, 250000, 120000],
    'CreditHistory': [1, 0, 0, 1, 0, 1, 0, 1],
    'CreditScore': [1, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('CreditScore', axis=1)
y = df['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.60, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

