
import numpy as np
from ml_from_scratch.supervised_learning.regression.linear_regression.models.linear_regression import LinearRegression


# 1. Prepare the training dataset
X = np.array([
    [500],
    [700],
    [900],
    [1100],
    [1300]
])

y = np.array([
    [25],
    [35],
    [45],
    [55],
    [65]
])


# 2. Create the Linear Regression model
model = LinearRegression(
    learning_rate=1e-7,
    n_iterations=1000
)


# 3. Train the model
model.fit(X, y)


# 4. Display the training results
print("\n--- Training Results ---")

print("\nFirst 10 loss values:")
print(model.loss_history[:10])

print("\nLast 10 loss values:")
print(model.loss_history[-10:])

print("\nLearned weight:")
print(model.w)

print("\nLearned bias:")
print(model.b)


# 5. Generate predictions
predictions = model.predict(X)

print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(y)


new_house = np.array([[1500]])

prediction = model.predict(new_house)

print("Predicted price:", prediction)