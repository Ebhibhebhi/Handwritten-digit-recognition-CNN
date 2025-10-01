## Description
This project implements a neural network from scratch in NumPy to classify handwritten digits from the MNIST dataset. Instead of relying on deep learning 
frameworks like TensorFlow or PyTorch, the entire pipeline (data loading, feedforward computation, backpropagation, and stochastic gradient descent) is coded 
manually. The goal is to deeply understand the mathematical foundations of neural networks while achieving high accuracy on a classic benchmark problem in 
machine learning.

## Motivation
As with most of us, the catalyst for my interest in neural networks was [this video](https://youtu.be/aircAruvnKk?si=4hzVh-wsm9fzu2Ss) by 3blue1brown, 
where he explains the mathematics behind how neural networks can perform classification tasks using the example of handwritten digits. I was fascinated by just how 
simple and elegant the mathematics was and therefore decided to implement the neuralnetwork myself, without using any DL frameworks so I could gain a deeper intuition 
behind the function of CNNs.

## How it works

This project implements a feed-forward neural network trained from scratch with mini-batch **stochastic gradient descent (SGD)** and **backpropagation** to classify MNIST digits.

### Pipeline
1. **Data loading** (`mnist_loader.py`)  
   - Images are reshaped into 784-dimensional column vectors (`28×28 → 784`) scaled to `[0,1]`.  
   - Labels are converted into one-hot vectors (`0–9` → `[0,0,...,1,...,0]`).  

2. **Network architecture** (`my_network.py`)  
   - Default: `784 → 70 → 10` fully connected layers.  
   - Parameters: weights and biases initialized randomly.  
   - Activation: **sigmoid** function.  

3. **Training loop** (`mnist_run.py`)  
   - Shuffle training set each epoch.  
   - Split into mini-batches (default: 10).  
   - For each batch: forward pass → backpropagation → parameter update.  

4. **Evaluation**  
   - On validation/test sets using `argmax` of the 10-dimensional output vector.  
   - Prints accuracy each epoch (e.g., `Epoch 0: 9120 / 10000`).  

### Forward pass
- Compute weighted sums: `z^l = W^l a^{l-1} + b^l`  
- Apply activation: `a^l = σ(z^l)` with `σ(z) = 1/(1+e^{-z})`  

### Backpropagation
- Output error: `δ^L = (a^L − y) ⊙ σ'(z^L)`  
- Hidden layers: `δ^l = (W^{l+1}ᵀ δ^{l+1}) ⊙ σ'(z^l)`  
- Gradients:  
  - `∂C/∂W^l = (1/m) Σ δ^l a^{l-1}ᵀ`  
  - `∂C/∂b^l = (1/m) Σ δ^l`

### Parameter update
- For learning rate `η`:  
  - `W^l ← W^l − η ∂C/∂W^l`  
  - `b^l ← b^l − η ∂C/∂b^l`

### Expected results
With default settings (`30` epochs, batch size `10`, learning rate `3`), the network typically achieves **>90% accuracy** on the MNIST test set after several epochs.

## Project structure & Data

├─ code/  
│ ├─ my_network.py # Core neural network: feedforward, SGD, backprop, evaluation  
│ ├─ mnist_loader.py # Loads and wraps MNIST (mnist.pkl.gz) for the network  
│ └─ mnist_run.py # Entry point: trains and evaluates the model  
├─ data/  
│ └─ mnist.pkl.gz # Pre-pickled MNIST dataset   
└─ README.md  

This project uses the MNIST dataset in a pre-pickled format (`mnist.pkl.gz`).  
Make sure the file is placed in the `data/` folder as shown above.  
The loader in `mnist_loader.py` is set up to look for the dataset in that location by default, so with this structure it will work out of the box.

If your data file is stored somewhere else, you can point the loader to it directly in `mnist_run.py`:

```python
training_data, validation_data, test_data = load_data_wrapper(path="path/to/mnist.pkl.gz")
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<Ebhibhebhi>/Handwritten-digit-recognition-CNN.git
   cd Handwritten-digit-recognition-CNN
2. **Install NumPy**
   ```bash
   pip install numpy

## How to Run

From the repo root, run:
  ```bash
    python -m code.mnist_run
  ```
By default, the script:

- **Loads MNIST** via `mnist_loader.py`  
- **Builds a network** with layers `[784, 70, 10]`  
- **Trains** with `epochs=30`, `mini_batch_size=10`, `learning_rate=3`  
- **Prints test accuracy** after each epoch  

**Example output:**  
`Epoch 0: 9120 / 10000`  
`Epoch 1: 9256 / 10000`  
`...`  
`Epoch 29: 9463 / 10000`  

With the default hyperparameters, the network typically reaches ~94–95% accuracy on the MNIST test set after training.
You can experiment by editing `epochs`, `batch size`, `learning rate`, or the hidden layer size in `mnist_run.py`.
