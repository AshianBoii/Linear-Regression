import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Define the file path where you saved the dataset locally
file_path = 'auto-mpg.data'

# Define the column names since the dataset doesn’t have a header
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

# Load the dataset from the local file
data = pd.read_csv(file_path, sep=r'\s+', names=columns, na_values='?')

# Drop the 'car_name' column 
data = data.drop('car_name', axis=1)

# Drop rows with missing values
data = data.dropna()

# Convert all columns to numeric 
data = data.apply(pd.to_numeric)

# Prepare features and target variable 
X = data.iloc[:, 1:].values  # columns 2-8
y = data['mpg'].values       # column 1: 'mpg's

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
coefficients = []
rmse_values = []

# Row labels 
index = [f'fold{i}' for i in range(1, 11)]  # Creates fold1 to fold10

# Exclude the intercept column
columns_with_rmse = columns[1:-1] + ['RMSE']  # ADD RMSE

# Perform cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compute beta using the normal equation
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    # Predict the test set
    y_pred = X_test @ beta

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    coefficients.append(beta)
    rmse_values.append(rmse)

# Create a DataFrame to store the results
results = []
for coef, rmse in zip(coefficients, rmse_values):
    results.append(np.append(coef, rmse))  # Append coefficients with RMSE

# Convert the results to a DataFrame
df = pd.DataFrame(results, columns=columns_with_rmse, index=index)

# Output the DataFrame showing coefficients and RMSE for each fold
print(df)