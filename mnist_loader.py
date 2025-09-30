# mnist_loader.py  (Python 3)
import gzip
import pickle
import numpy as np

# Adjust this path to where *you* put the file.
# In the book repo the code directory sits beside a "data" folder:
#     code/
#     data/mnist.pkl.gz
DEFAULT_PATH = "../data/mnist.pkl.gz"

def load_data(path: str = DEFAULT_PATH):
    """Return (training_data, validation_data, test_data) as in the book."""
    with gzip.open(path, "rb") as f:
        # File was pickled in Python 2; need encoding for Python 3:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

def vectorized_result(j: int) -> np.ndarray:
    e = np.zeros((10, 1), dtype=np.float32)
    e[j] = 1.0
    return e

def load_data_wrapper(path: str = DEFAULT_PATH):
    """
    Return (training_data, validation_data, test_data) where:
      training_data = list[(x(784,1), y(10,1))]
      validation/test = list[(x(784,1), y_int)]
    Shapes and formats match what the book's Network expects.
    """
    tr_d, va_d, te_d = load_data(path)

    training_inputs  = [np.reshape(x, (784, 1)).astype(np.float32) for x in tr_d[0]]
    training_results = [vectorized_result(int(y)) for y in tr_d[1]]
    training_data    = list(zip(training_inputs, training_results))  # list(...)

    validation_inputs = [np.reshape(x, (784, 1)).astype(np.float32) for x in va_d[0]]
    validation_data   = list(zip(validation_inputs, [int(y) for y in va_d[1]]))

    test_inputs = [np.reshape(x, (784, 1)).astype(np.float32) for x in te_d[0]]
    test_data   = list(zip(test_inputs, [int(y) for y in te_d[1]]))

    return training_data, validation_data, test_data