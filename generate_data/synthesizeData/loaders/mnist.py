import gzip
import os
import struct
from array import array

import numpy as np


class DataLoader(object):
    def __init__(self, *inputs):
        pass
    
    def get_batch(self, batch_size):
        pass


def load_mnist_file(image_path, label_path):
    """Load the mnist file

    # Arguments
        image_path [str]: the path to file
        label_path [str]: the path to label
    
    # Returns
        [np array]: the image
        [np array]: the label
    """
    with gzip.open(image_path, 'rb') as f_in:
        magic, size, rows, cols = struct.unpack('>IIII', f_in.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'
                             .format(magic))
        
        image_data = np.array(f_in.read())
        image_data = image_data.reshape(-1, rows * cols)
        

def mnist_loader(folder_path):
    """Return the MNIST data loader

    # Arguments
        folder_path [str]: the folder containing MNIST files
    
    # Returns
        [loader]: the data loader
    """
    files = [
        os.path.join(folder_path, 'train-images-idx3-ubyte.gz'),
        os.path.join(folder_path, 'train-labels-idx1-ubyte.gz'),
        os.path.join(folder_path, 't10k-images-idx3-ubyte.gz'),
        os.path.join(folder_path, 't10k-labels-idx1-ubyte.gz')
    ]

    image_sets = []
    label_sets = []


    return 


if __name__ == '__main__':
    loader = mnist_loader(
        '/Users/ducprogram/Documents/WorkingDir/deep-learning/learning/nn/cnn/mnist'
    )
    loader.get_batch()