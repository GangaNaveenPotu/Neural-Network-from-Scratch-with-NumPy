import numpy as np
from src.data_loader import load_mnist
from src.neural_network import NeuralNetwork
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test = load_mnist()

    split_index = int(0.9 * X_train.shape[0])
    X_train, X_val = X_train[:split_index], X_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    layer_sizes = [X_train.shape[1], 128, 64, y_train.shape[1]]
    
    nn = NeuralNetwork(layer_sizes, learning_rate=0.001)

    print("Training the neural network with learning rate 0.001...")
    history = nn.train(X_train, y_train, epochs=100, batch_size=64, X_val=X_val, y_val=y_val, early_stopping_patience=10)

    print("Evaluating the neural network...")
    predictions = nn.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss (lr=0.001)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy (lr=0.001)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_history_lr_001.png')

if __name__ == '__main__':
    main()