import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('advertising.csv')

# Display basic information and first few rows
print('Dataset Info:')
df.info()
print('\nFirst 5 rows:')
print(df.head())

print('\nDescriptive statistics:')
print(df.describe())

print('\nMissing values:')
print(df.isnull().sum())

# Separate features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print('\nFeatures (X) shape:', X.shape)
print('Target (y) shape:', y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nX_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print('\nLinear Regression Model Trained.')
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nModel Evaluation:')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Visualize actual vs. predicted sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.savefig('actual_vs_predicted_sales.png')
plt.show()

print('\nVisualization saved as actual_vs_predicted_sales.png')
