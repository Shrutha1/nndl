import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def fit(self, x, y):
        self.weights, self.bias = np.zeros(x.shape[1]), 0
        for _ in range(self.epochs):
            for idx, sample in enumerate(x):
                y_pred = self.sigmoid(np.dot(sample, self.weights) + self.bias)
                error = y[idx] - y_pred
                update = self.learning_rate * error * y_pred * (1 - y_pred)
                self.weights += update * sample
                self.bias += update

# AND gate data
x_and, y_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])

# Train perceptron for AND gate
perceptron_and = Perceptron(learning_rate=0.1, epochs=1000)
perceptron_and.fit(x_and, y_and)

print("AND gate predictions:")
for i in range(len(x_and)):
    pred = 1 if perceptron_and.predict(x_and[i]) >= 0.5 else 0
    print(f"Input: {x_and[i]} - Prediction: {pred}")

# OR gate data
x_or, y_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])

# Train perceptron for OR gate
perceptron_or = Perceptron(learning_rate=0.1, epochs=1000)
perceptron_or.fit(x_or, y_or)

print("\nOR gate predictions:")
for i in range(len(x_or)):
    pred = 1 if perceptron_or.predict(x_or[i]) >= 0.5 else 0
    print(f"Input: {x_or[i]} - Prediction: {pred}")
