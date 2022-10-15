import numpy as np

architecture = np.load('outputs/architectures.npy')
print(architecture.shape)
#print(architecture)

binary_architecture = np.load('outputs/binary_architectures.npy')
print(binary_architecture.shape)
print(binary_architecture)