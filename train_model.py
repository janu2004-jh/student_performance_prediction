import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Create model folder
if not os.path.exists("model"):
    os.makedirs("model")

# Load dataset
df = pd.read_csv("student_dataset_100k.csv")

# Keep only required columns
df = df[["Study_Hours", "Attendance", "Previous_Marks", "Final_Result"]]

# Convert target
df["Final_Result"] = df["Final_Result"].map({"Pass": 1, "Fail": 0})

# Features & target
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")