import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = {}
        self.grads = {}
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2 / self.layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, self.layer_sizes[i]))
    def _relu(self, Z):
        return np.maximum(0, Z)

    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def forward(self, X):
        self.cache = {}
        A = X
        self.cache['A0'] = X

        for i in range(1, len(self.layer_sizes) - 1):
            Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
            A = self._relu(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A

        i = len(self.layer_sizes) - 1
        Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
        A = self._softmax(Z)
        self.cache[f'Z{i}'] = Z
        self.cache[f'A{i}'] = A

        return A

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def backward(self, X, y_true):
        m = X.shape[0]
        num_layers = len(self.layer_sizes) - 1
        y_pred = self.cache[f'A{num_layers}']
        
        dZ = y_pred - y_true
        dW = np.dot(self.cache[f'A{num_layers-1}'].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        self.grads[f'dW{num_layers}'] = dW
        self.grads[f'db{num_layers}'] = db

        for i in range(num_layers - 1, 0, -1):
            dZ_prev = dZ
            W_prev = self.params[f'W{i+1}']
            dZ = np.dot(dZ_prev, W_prev.T) * self._relu_derivative(self.cache[f'Z{i}'])
            dW = np.dot(self.cache[f'A{i-1}'].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.grads[f'dW{i}'] = dW
            self.grads[f'db{i}'] = db

    def update_params(self):
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] -= self.learning_rate * self.grads[f'dW{i}']
            self.params[f'b{i}'] -= self.learning_rate * self.grads[f'db{i}']

    def train(self, X_train, y_train, epochs, batch_size=64, X_val=None, y_val=None, early_stopping_patience=None, early_stopping_threshold=1e-4):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        m = X_train.shape[0]

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation, :]
            y_shuffled = y_train[permutation, :]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size, :]
                y_batch = y_shuffled[i:i+batch_size, :]

                self.forward(X_batch)

                self.backward(X_batch, y_batch)

                self.update_params()

            y_pred_full = self.forward(X_train)
            loss = self.compute_loss(y_train, y_pred_full)
            predictions = self.predict(X_train)
            true_labels = np.argmax(y_train, axis=1)
            accuracy = np.mean(predictions == true_labels)

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(y_val, y_pred_val)
                val_predictions = self.predict(X_val)
                val_true_labels = np.argmax(y_val, axis=1)
                val_accuracy = np.mean(val_predictions == val_true_labels)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss - early_stopping_threshold:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return history

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)