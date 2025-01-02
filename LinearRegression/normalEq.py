import numpy as np
from sklearn.datasets import make_regression

# Create data set.
X, y = make_regression(n_samples=100, n_features=1,
					n_informative=1, noise=10, random_state=10)

def linear_regression_normal_equation(X, y):
	X_transpose = np.transpose(X)
	X_transpose_X = np.dot(X_transpose, X)
	X_transpose_y = np.dot(X_transpose, y)
	
	try:
		theta = np.linalg.solve(X_transpose_X, X_transpose_y)
		return theta
	except np.linalg.LinAlgError:
		return None


# Add a column of ones to X for the intercept term
X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

theta = linear_regression_normal_equation(X_with_intercept, y)
if theta is not None:
	print(theta)
else:
	print("Unable to compute theta. The matrix X_transpose_X is singular.")
# To predict on new Data Instances 

def predict(X, theta):
    predictions = np.dot(X, theta)
    return predictions


# Input features for testing
X_test = np.array([[1],[4]])
X_test_with_intercept = np.c_[np.ones((X_test.shape[0],1)), X_test]
predictions = predict(X_test_with_intercept, theta)
print("Predictions:", predictions)