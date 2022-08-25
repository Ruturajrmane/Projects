import sys
from os import path 
import torch

sys.path.insert(0, r'C:\Users\RuturajMane\Desktop\MNIST CNN\mnist-ruturaj\src')
from model import inimodel

def test_model_save_load():
    """
    Tests for the model saving process
    """
    model = inimodel()
        
    #=================================
    # TEST SUITE
    #=================================
    # Check the model file is created/saved in the directory
    assert path.exists(r'C:\Users\RuturajMane\Desktop\MNIST CNN\mnist-ruturaj\Models')
    # Check that the model file can be loaded properly 
    # (by type checking that it is a sklearn linear regression estimator)
    torch.save({'model_state': model.state_dict()},r'C:\Users\RuturajMane\Desktop\MNIST CNN\mnist-ruturaj\Models\demo.sav')
    loaded_model = torch.load(r'C:\Users\RuturajMane\Desktop\MNIST CNN\mnist-ruturaj\Models\demo.sav')
    assert isinstance(loaded_model, dict)