import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("sales_data.csv")  # future_sales column
X = data.drop("future_sales", axis=1)
y = data["future_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Predicted Sales:", model.predict(X_test[:5]))
