import pytest
# importing sys
import sys
import torch
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r'C:\Users\RuturajMane\Desktop\MNIST CNN\mnist-ruturaj\src')
from model import inimodel
# To do not to use actual model for testing
import numpy as np


def data():
    return np.random.rand(2, 1,28,28) # Batch size 2

def test_model_output():
    model = inimodel()
    image = torch.from_numpy(data()).float()
    pred = model(image)
    # Output need to be in the form of tensor 
    assert torch.is_tensor(pred), 'Output expected tensor'
    # As two batches are given the 
    assert len(pred) == 2
    # Need to give probabilities for 10 classe
    assert len(pred[0]) == 10


