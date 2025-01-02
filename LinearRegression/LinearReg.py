import numpy as np

class LinearRegressor:
    def __init__(self, x, y, alpha=0.01, epochs=1000):
        self.x = np.array(x)
        self.y = np.array(y)
        self.alpha = alpha
        self.epochs = epochs
        self.b0 = 0
        self.b1 = 0

    def fit(self):
        n = len(self.x)
        for _ in range(self.epochs):
            y_pred = self.b0 + self.b1 * self.x
            d_b0 = (-2/n) * sum(self.y - y_pred)
            d_b1 = (-2/n) * sum(self.x * (self.y - y_pred))
            self.b0 -= self.alpha * d_b0
            self.b1 -= self.alpha * d_b1

    def predict(self, x):
        return self.b0 + self.b1 * x

def update_coeff(value):
    # Assuming this function updates some global or class-level coefficients
    pass

# Corrected code snippet
update_coeff(1)
linearRegressor = LinearRegressor(
    x=[i for i in range(12)],
    y=[2 * i + 3 for i in range(12)],
    alpha=0.03
)
linearRegressor.fit()
print(linearRegressor.predict(20))