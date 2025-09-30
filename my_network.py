import numpy as np
import random


class Network(object):
    def __init__(self, sizes): # You enter a list of numbers as the argument where each number corresponds to the number of neurons in that layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """Return the output of the network if a is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """Train the neural network using mini-batch stochastic gradient descent. The
         "training_data" is a list of tuples "(x,y)" representing the training inputs and the
         desired outputs. The other non-optional parameters are self-explanatory. if "test_data"
         is provided then the network will be evaluated against the test data after each epoch,
         and partial progress is printed out. This is useful for tracking progress, but slows things
         down substantially."""
        
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs): # fixed
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
       
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropogation
        to a single minibatch. The "mini_batch" is a list of tuples "(x,y)", and "eta" is the learning
        rate."""

        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) # returns the gradients of the loss of that specific training example wrt every weight and bias in the network
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # running sum of the gradients of every single example cost function wrt all weights
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # running sum of the gradients of every single example cost function wrt all biases
        
        # updating the weights and biases using the worb_new = worb - (eta/m)*sum_gradients
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] 
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Returns (nabla_b, nabla_w) = gradients of the single-example cost C_{x,y}
        wrt each bias and weight in the network. Shapes match self.biases and self.weights."""

        # gradient accumulators per layer
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        # forward pass: Store all z's and activations
        activation = x # (n_in, 1)
        activations = [x] # list of column vectors a^[0], a^[1], ...
        zs = [] # list of preactivations z = W a + b

        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b # matrix multiplication + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass: Output layer error = δ^L = ∂C/∂a^L ⊙ σ'(z^L)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T

        # Backpropagate to earlier layers:
        # for l = 2,3,...,num_layers-1:
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # δ^l = (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l)
            delta = (self.weights[-l + 1].T @ delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l - 1].T

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural networks outputs the correct result.
        Note that the neural network's output is the index of whichever neuron in the final layer has
        the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives for the output activations."""
        return output_activations - y

def sigmoid(z):
    """Sigmoid function"""
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))