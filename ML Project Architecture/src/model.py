from Dataclasses.layer import Layer
import json
import torch
from modelparser import ModelParser
from Dataclasses.hyperpar import Hyperparameters

class MnistModel(torch.nn.Module):
    def __init__(self,layer_list):
        super().__init__()
        self.layer_list = layer_list

    def _activation_mapper(self, act_string):

        if act_string == "ReLU":
            return torch.nn.ReLU()
        
        elif act_string == 'Sigmoid':
            return torch.nn.Sigmoid()
        
        else:
            pass

    def build_model(self):

        module_list = list()

        for layer_ix, layer in enumerate(self.layer_list):
            
            if "Linear" in layer.Layer_name:
                module_list.append(torch.nn.Flatten())
                intilayer = torch.nn.Linear(layer.Num_inputs, layer.Num_outputs, bias = layer.Bias)
            if "Conv" in layer.Layer_name:
                intilayer = torch.nn.Conv2d(in_channels = layer.Num_inputs, out_channels = layer.Num_outputs, kernel_size = layer.kernel_size,
                stride = layer.stride, padding = layer.padding, bias = False)
            if "MaxPool2d" in layer.Layer_name:
                intilayer = torch.nn.MaxPool2d(kernel_size = layer.kernel_size)
            
            module_list.append(intilayer)

            act = self._activation_mapper(layer.Activation)
            if act != None:
                module_list.append(act)

            dpt = torch.nn.Dropout2d(layer.dropout) if layer.Dropout else False
            if dpt:
                module_list.append(dpt)

        print(module_list)
        self.pred = torch.nn.Sequential(*module_list)

        return self.pred


if __name__ == "__main__":
    parser = ModelParser("../base_config.json")
    layers = parser.get_list()
    kwargs = parser.get_hp()[0]
    model = MnistModel(layers)
    model = model.build_model()
    Hyperparameters(**kwargs)

    