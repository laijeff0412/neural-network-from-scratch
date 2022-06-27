import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_derivative(Z):
    return Z > 0  # if Z>0 its slope=1(True)


def softmax(Z):  # 應用在輸出層
    A = np.exp(Z) / sum(np.exp(Z))
    return A.T


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


def gradient_descent(A, Y, Z, X, weights, layers_count, nr):  # 計算梯度
    one_hot_Y = one_hot(Y)
    pZ2 = A[len(A) - 1] - one_hot_Y.T
    pW2 = pZ2.dot(A[len(A) - 2].T) / nr
    pb2 = np.sum(pZ2) / nr

    pZ1 = weights[len(weights) - 1].T.dot(pZ2) * ReLU_derivative(Z[len(Z) - 2])
    pW1 = pZ1.dot(X.T) / nr
    pb1 = np.sum(pZ1) / nr

    return [pW1, pW2], [pb1, pb2]


def update_params(weights, biases, gradient_weights, gradient_biases, alpha):
    updated_weights = []
    updated_biases = []
    for i in range(len(weights)):
        updated_weights.append(weights[i] - alpha * gradient_weights[i])
        updated_biases.append(biases[i] - alpha * gradient_biases[i])
    return updated_weights, updated_biases


class NN:
    def __init__(self, *layers):
        self.layers = layers
        self.params = self.generate_params()
        self.layers_count = int(len(self.layers))

    def generate_params(self):
        layers_count = int(len(self.layers))
        for i in range(1, layers_count):
            globals()['W' + str(i)] = np.random.rand(self.layers[i], self.layers[i - 1]) - 0.5
            globals()['b' + str(i)] = np.random.rand(self.layers[i], 1) - 0.5
        return [globals()['W' + str(i)] for i in range(1, layers_count)], [globals()['b' + str(i)] for i in
                                                                           range(1, layers_count)]

    def forward_prop(self, params, X):
        weights = params[0]
        biases = params[1]
        Z = []
        A = [X]
        for i in range(1, self.layers_count - 1):  # the last layer use softmax instead of ReLU
            Z.append(weights[i - 1].dot(A[i - 1]) + biases[i - 1])
            A.append(ReLU(Z[i - 1]))
        Z.append(weights[self.layers_count - 2].dot(A[self.layers_count - 2]) + biases[self.layers_count - 2])
        A.append(softmax(Z[self.layers_count - 2]).T)
        return weights, biases, Z, A[1:]

    def backward_prop(self, A, Y, Z, X, weights, biases, alpha):  # chain rule, lots of partial derivative
        gradient_weights, gradient_biases = gradient_descent(A, Y, Z, X, weights, self.layers_count, nr)
        updated_weights, updated_biases = update_params(weights, biases, gradient_weights, gradient_biases, alpha)
        return [updated_weights, updated_biases]


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def train(model, X, Y, iteration, alpha):  # specific for the mnist dataset
    # the main training loop
    weights, biases, Z, A = model.forward_prop(model.params, X)  # run the randomly generated params first
    for i in range(iteration):
        updated_params = model.backward_prop(A, Y, Z, X, weights, biases, alpha)
        weights, biases, Z, A = model.forward_prop(updated_params, X)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A)
            print(get_accuracy(predictions, Y))

    return updated_params


def get_predictions(A):
    A = A[len(A) - 1]
    return np.argmax(A, 0)


def make_predictions(X, params):
    _, _, _, A = model.forward_prop(params, X)
    predictions = get_predictions(A)
    return predictions


def test_prediction(index, params):
    current_image = X_val[:, index, None]
    prediction = make_predictions(X_val[:, index, None], params)
    label = Y_val[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def validation(params):
    for i in range(100):
        test_prediction(i, params)


if __name__ == '__main__':
    data = pd.read_csv("train.csv")  # read mnist dataset
    data = np.array(data)
    nr, nc = data.shape
    # training data
    data_train = data[1000:nr].T
    Y_train = data_train[0]  # labels
    X_train = data_train[1:nc] / 255.  # inputs
    # validation data
    data_val = data[0:1000].T
    Y_val = data_val[0]  # labels
    X_val = data_val[1:nc] / 255.  # inputs

    model = NN(784, 10, 10)  # input_layer, *hidden_layer, output_layer
    # train
    updated_params = train(model, X_train, Y_train, 300, 0.10)
    # validation
    validation(updated_params)
