import unittest
import numpy as np
from src.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.layer_sizes = [2, 3, 2]
        self.nn = NeuralNetwork(self.layer_sizes)
        self.X = np.array([[0.1, 0.2]])
        self.y = np.array([[1, 0]])

    def test_forward_pass(self):
        output = self.nn.forward(self.X)
        self.assertEqual(output.shape, (1, 2))
        self.assertAlmostEqual(np.sum(output), 1.0)

    def test_backward_pass(self):
        y_pred = self.nn.forward(self.X)
        self.nn.backward(self.X, self.y)
        analytical_grads = self.nn.grads

        epsilon = 1e-5
        numerical_grads = {}
        for key, param in self.nn.params.items():
            numerical_grads[f'd{key}'] = np.zeros_like(param)
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i, j] += epsilon
                    y_pred_plus = self.nn.forward(self.X)
                    loss_plus = self.nn.compute_loss(self.y, y_pred_plus)

                    param[i, j] -= 2 * epsilon
                    y_pred_minus = self.nn.forward(self.X)
                    loss_minus = self.nn.compute_loss(self.y, y_pred_minus)

                    numerical_grads[f'd{key}'][i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                    param[i, j] += epsilon

        for key in analytical_grads.keys():
            self.assertTrue(np.allclose(analytical_grads[key], numerical_grads[key], rtol=1e-5, atol=1e-5))

if __name__ == '__main__':
    unittest.main()