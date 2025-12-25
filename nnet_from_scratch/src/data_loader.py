import numpy as np
from mnist import MNIST
from sklearn.preprocessing import OneHotEncoder

def load_mnist(mnist_path='../MNIST'):
    mndata = MNIST(mnist_path)
    mndata.gz = False
    
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0
    
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_mnist()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")