# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
data = pd.read_csv("diabetes.csv")
print(data.head())
# Step 3: Understand the data
print(data.info())
print(data.describe())
columns = ['Glucose', 'BloodPressure', 'BMI']
data[columns] = data[columns].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})
print(feature_importance.sort_values(by='Importance', ascending=False))