import json
import os

class Layer:
    def __init__(self, **kwargs):
        self._attr_dict = kwargs
        self.Layer_name = kwargs["Layer_name"]
        self.Layer_idx = kwargs["Layer_idx"]
        self.Num_inputs = kwargs["Num_inputs"]
        self.Num_outputs = kwargs ["Num_outputs"]
        self.Activation = kwargs["Activation"]
        self.Bias = kwargs["Bias"]
        self.Dropout = kwargs["Dropout"]
        self.kernel_size = kwargs["kernel_size"]
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]

    def print_info(self):
        print(self._attr_dict)