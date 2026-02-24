from statistics import mean

import numpy as np


class Perceptron:
    def __init__(
        self,
        n_weights=0,
        learning_rate=0.01,
        activation_function="sigmoid",
        random_state=None,
    ):
        """
        the activation functions available :
            - sigmoid
            - relu
            - binary_step
        """
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.random.rand(n_weights)
        self.bias = np.random.rand(1)

    def forward(self, input):
        return self.activation(self.weights @ input + self.bias)

    def fit(self, X, y, epochs=1000):
        print("Training the perceptron...")
        for epoch in range(epochs):
            for i in range(len(X)):
                output = self.forward(X[i])
                error = y[i] - output
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
            print(
                "current epoch:",
                epoch + 1,
                "error:",
                mean([abs(y[i] - self.forward(X[i])) for i in range(len(X))]),
            )

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            return max(0, x)
        elif self.activation_function == "binary_step":
            return 1 if x >= 0 else 0
        else:
            raise ValueError("Unsupported activation function")
