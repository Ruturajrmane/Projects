import pytest
from unittest.mock import patch
# importing sys
import sys
import torch
import numpy as np


# adding Folder_2/subfolder to the system path
sys.path.insert(0, r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src')
from model import inimodel
model = inimodel()


def test_model_output():
    image = torch.from_numpy(np.random.rand(2, 1, 28,28)).float()
    pred = model(image)
    # Output need to be in the form of tensor 
    assert torch.is_tensor(pred), 'Output expected tensor'
    # As two batches are given the 
    assert len(pred) == 2
    # Need to give probabilities for 10 classe
    assert len(pred[0]) == 10

@pytest.mark.parametrize("a,b,c,d", [[1,29,28, 0], [4,8,0,10]])
def test_negative_input(a,b,c,d):
    image = torch.from_numpy(np.random.rand(a,b,c,d)).float()
    with pytest.raises(RuntimeError):
        pred = model(image)
    











