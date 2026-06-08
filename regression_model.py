import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # now this linear regression is for regression problems, not classification problems
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
X = np.random.rand(200, 1)  # Input feature
y = 2 + 3 * X + np.random.randn(200, 1)  # True underlying relationship + random noise

# Split the data into training and testing sets
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train , X_test = train_test_split(X, test_size = 0.2 , random_state = 42)
y_train , y_test = train_test_split(y, test_size = 0.2 , random_state = 42)
# Create linear regression model

model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Squared Error:', mse) 
print(f'Root Mean Squared Error: {rmse:.2f}')
print('Mean Absolute Error:', mae)
print('R-squared:', r2)