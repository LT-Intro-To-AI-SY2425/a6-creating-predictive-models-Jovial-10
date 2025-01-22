import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv(r"C:\Users\jbrana@cps.edu\a6-creating-predictive-models-Jovial-10\Proj-6\cars_data.csv")


# Define features (X) and target (y)
X = data[["Car Model", "Miles Per Gallon", "Cylinders", "Displacement", 
          "Horsepower", "Weight", "Acceleration", "Year", "Origin", "Quarter Mile", "Gears"]]
y = (data["Miles Per Gallon"] > 20).astype(int)  # Target: High MPG cars (1 = MPG > 20)

# Encode categorical features
encoder = OneHotEncoder(sparse=False)
encoded_cars = encoder.fit_transform(X[["Car Model", "Origin"]])

# Standardize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[["Miles Per Gallon", "Cylinders", "Displacement", 
                                          "Horsepower", "Weight", "Acceleration", 
                                          "Year", "Quarter Mile", "Gears"]])

# Combine encoded and scaled features
import numpy as np
X_processed = np.hstack((scaled_features, encoded_cars))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Test prediction on new data
new_data = pd.DataFrame([["Toyota Corolla", 33.9, 4, 71.1, 65, 4.22, 1.835, 19.9, 1, 1, 4]],
                        columns=["Car Model", "Miles Per Gallon", "Cylinders", "Displacement", 
                                 "Horsepower", "Weight", "Acceleration", "Year", 
                                 "Origin", "Quarter Mile", "Gears"])

# Encode and scale the new data
encoded_new_cars = encoder.transform(new_data[["Car Model", "Origin"]])
scaled_new_features = scaler.transform(new_data[["Miles Per Gallon", "Cylinders", "Displacement", 
                                                 "Horsepower", "Weight", "Acceleration", 
                                                 "Year", "Quarter Mile", "Gears"]])

new_data_processed = np.hstack((scaled_new_features, encoded_new_cars))

# Predict on the new data
y_pred_new = model.predict(new_data_processed)
print(f"Prediction for new data: {'High MPG' if y_pred_new[0] == 1 else 'Low MPG'}")
