import pytest
# importing sys
import sys
import torch
 # adding Folder_2/subfolder to the system path
sys.path.insert(0, r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src')
from model import inimodel

def test_parsing(): 
    model = inimodel()
    # Ensuring that configrations of architecture are parsed correctly and formed model 
    assert torch.is_tensor(model[0].weight)








