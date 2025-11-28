import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- 1. Load the Iris dataset ---
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# --- 2. Explore and visualize the data (Optional: run in a suitable environment like Jupyter for plots) ---
# sns.pairplot(iris_df, hue='species', markers=["o", "s", "D"])
# plt.show()
# sns.heatmap(iris_df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# --- 3. Split the dataset into training and testing sets ---
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train a model using a classifier ---
log_reg = LogisticRegression(solver='liblinear', multi_class='auto')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# --- 5. Evaluate model accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix, "\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# --- Visualize the confusion matrix (Optional) ---
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for Iris Classification')
# plt.show()
