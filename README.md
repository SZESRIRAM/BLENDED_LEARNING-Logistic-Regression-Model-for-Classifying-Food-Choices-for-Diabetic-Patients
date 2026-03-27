# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the dataset, and preprocess the data by scaling features and encoding target labels.<br/>
2. Split the dataset into training and testing sets.<br/>
3. Create and train the Logistic Regression model using the training data.<br/>
4. Predict the test data and evaluate the model using accuracy, classification report, and confusion matrix.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("food_items.csv")

# Inspect the dataset
print("Name:Ponsriram P ")
print("Reg. No:212225240105 ")
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Model parameters
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000

# Define logistic regression model
l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class,
    solver=solver,
    max_iter=max_iter
)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```

## Output:
<img width="847" height="656" alt="image" src="https://github.com/user-attachments/assets/0a0879b8-c200-4a09-a7e4-4f8cc62a1e25" />

<img width="733" height="650" alt="image" src="https://github.com/user-attachments/assets/4046654b-aebb-4a54-ac0c-ec344a9e2c93" />
<img width="591" height="316" alt="image" src="https://github.com/user-attachments/assets/bea1f4ba-75dd-4279-beaf-1009aac7bed0" />




## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
