import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from model import ModelParser
from model import MnistModel
from torch.utils.data import DataLoader

test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor()) # List of the tuples with tuples having pixel and label
len(test_dataset)
j=0
test = DataLoader(test_dataset, batch_size=1)

# Model initializing

parser = ModelParser("../base_config.json")
layers = parser.get_list()
kwargs = parser.get_hp()[0]
model = MnistModel(layers)
model = model.build_model()

# loading the model

model.load_state_dict((torch.load('../Models/MNIST_50_30_0.0001_Adam_cross_entropy.pth'))['model_state'])
print(model)
# Testing the model 
model.eval()
with torch.no_grad():
    for image,label in test:
        pred = model(image)
        _, pred = torch.max(pred, dim = 1)
    #     print(pred)
    #     print(label)
        if pred == label:
            j+=1
    print('Test dataset accauracy :', j/len(test_dataset))


