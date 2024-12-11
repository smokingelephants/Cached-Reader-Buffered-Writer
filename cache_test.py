from cached_read_buffered_write_test import CachedReaderBufferedWriter
import numpy as np 

class SimpleRegressionNN:
    def __init__(self, input_dim, lr=0.01):
        """Initialize a simple regression neural network."""
        self.input_dim = input_dim
        self.lr = lr
        self.weights = np.random.randn(input_dim)  # Randomly initialize weights
        self.bias = np.random.randn()  # Randomly initialize bias

    def predict(self, input_data):
        """Perform a forward pass."""
        input_data = np.array(input_data)
        return np.dot(input_data, self.weights) + self.bias

    def train(self, inputs, targets):
        """Perform a simple gradient descent update."""
        predictions = np.dot(inputs, self.weights) + self.bias
        errors = targets - predictions

        # Compute gradients
        grad_weights = -2 * np.dot(errors, inputs) / len(inputs)
        grad_bias = -2 * np.mean(errors)

        # Update parameters
        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias

# Usage Example
if __name__ == "__main__":
    # Create the parent super class with cache and buffer
    parent = CachedReaderBufferedWriter(cache_size=5, buffer_size=3, cache_expiry=3)

    # Create a child neural network and attach it to the parent
    child_nn = SimpleRegressionNN(input_dim=2, lr=0.01)
    parent.set_child(child_nn)

    # Query the network multiple times
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration

    print("Querying:")
    for _ in range(5):
        print([1.0, 2.0], parent.query([1.0, 2.0]))  # Perform prediction and manage cache expiration
    #    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
        print([3.0, 4.0], parent.query([3.0, 4.0]))  # Perform prediction and manage cache expiration

    # Buffer updates and perform batched training
    print("Updating:")
    parent.update([1.0, 2.0], 5.0)
    parent.update([2.0, 3.0], 10.0)
    parent.update([3.0, 4.0], 15.0)  # This triggers a batched update


    # Query the network multiple times
    print("Querying:")
    for _ in range(5):
        print([1.0, 2.0], parent.query([1.0, 2.0]))  # Perform prediction and manage cache expiration
        #print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
        print([3.0, 4.0], parent.query([3.0, 4.0]))  # Perform prediction and manage cache expiration

    # Buffer updates and perform batched training
    print("Updating:")
    for _ in range(5):
        parent.update([1.0, 2.0], 5.0)
        parent.update([2.0, 3.0], 10.0)
        parent.update([3.0, 4.0], 15.0)  # This triggers a batched update

    # Query the network multiple times
    print("Querying:")
    #print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration

    for _ in range(5):
        print([1.0, 2.0], parent.query([1.0, 2.0]))  # Perform prediction and manage cache expiration
        #print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
        print([3.0, 4.0], parent.query([3.0, 4.0]))  # Perform prediction and manage cache expiration

    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    print([2.0, 3.0], parent.query([2.0, 3.0]))  # Perform prediction and manage cache expiration
    