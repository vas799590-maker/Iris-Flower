import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Collect Data ---
print("--- Step 1: Loading Data ---")
# The Boston housing dataset is often deprecated; using California housing dataset instead.
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
# Add the target variable (price in $100k) to the DataFrame
df['MedHouseVal'] = california.target
print(f"Data loaded with shape: {df.shape}\n")
# print(df.head()) # Uncomment to see the first few rows of data

# Define features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Clean and Preprocess the Data ---
print("--- Step 2: Preprocessing and Scaling Data ---")
# The California dataset is clean (no missing values), but we apply feature scaling.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled successfully.\n")

# --- Step 3: Exploratory Data Analysis (EDA) ---
print("--- Step 3: Performing EDA (Displaying Correlation Heatmap) ---")
# Calculate correlation matrix
corr_matrix = df.corr()

# Plot the heatmap (uncomment the plotting section to visualize)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix of Housing Features and Price')
# plt.show()
print("EDA complete. Correlation heatmap generated (visualization requires uncommenting plot code).\n")

# --- Step 4: Train a Linear Regression Model ---
print("--- Step 4: Training Linear Regression Model ---")
# Initialize the model
model = LinearRegression()

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)
print("Model training complete.\n")

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# --- Step 5: Evaluate Performance ---
print("--- Step 5: Evaluating Model Performance ---")
# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse) # RMSE is in the same units as the price
r2 = r2_score(y_test, predictions)

# Print the results
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R2) Score: {r2:.4f}')

# Visualize actual vs predicted prices (uncomment the plotting section to visualize)
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, predictions, alpha=0.5)
# plt.xlabel('Actual Prices ($100k)')
# plt.ylabel('Predicted Prices ($100k)')
# plt.title('Actual vs Predicted House Prices')
# Plot a perfect prediction line (y=x)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
# plt.show()
print("\nEvaluation complete. Scatter plot generated (visualization requires uncommenting plot code).")
