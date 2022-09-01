import pytest
from unittest.mock import patch
# importing sys
import sys
import numpy as np


# adding Folder_2/subfolder to the system path
sys.path.insert(0, r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src')
from train import preprocess

def test_preprocess():
    path = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src\data'

    train_load, val_load = preprocess(164, path)
    assert isinstance(train_load, object)
    assert isinstance(val_load, object)










