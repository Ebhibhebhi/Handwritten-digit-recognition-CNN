import mnist_loader, my_network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = my_network.Network([784, 100, 10])

net.SGD(training_data, 30, 10, 3, test_data)

