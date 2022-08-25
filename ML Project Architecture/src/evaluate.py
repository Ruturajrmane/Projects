import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import numpy as np
from model import inimodel
import argparse
from PIL import Image
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--outputpath', help = 'Enter the output path')
parser.add_argument('--image', help= 'Insert the image')
parser.add_argument('--label', help= 'Enter the label')
parser.add_argument('--modelpath', help='Enter the model path')
parser.add_argument('--testdataset', help = 'Enter the test dataset path')

args = parser.parse_args()

# Model initializing

model = inimodel()

# loading the model

if args.modelpath == None:
    path = '../Models/MNIST_50_30_0.0001_Adam_cross_entropy.pth'

else:
    path = args.modelpath

model.load_state_dict((torch.load(path))['model_state'])


def default(path):
    path = path + '/'
    test_dataset = MNIST(root= path, 
                        train=False,
                        transform=transforms.ToTensor()) # List of the tuples with tuples having pixel and label
    test = DataLoader(test_dataset, batch_size=1)
    # Testing the model
    j=0
    labels = []
    predict = []
    model.eval()
    with torch.no_grad():
        
        for image,label in test:
            pred = model(image)
            _, pred = torch.max(pred, dim = 1)
        #     print(pred)
        #     print(label)
            labels.append(label.item())
            predict.append(pred.item())

        precision, recall, fscore, support = score(labels, predict)

        df = pd.DataFrame(np.array([precision, recall, fscore, support]), index=['precision', 'recall', 'fscore', 'support'], columns= range(0,10))
        df['Average'] = [sum(precision)/len(precision),sum(recall)/len(recall),sum(fscore)/len(fscore),sum(support)/len(support) ]
    if args.outputpath != None:
        pathout = args.outputpath
    
    else:
        pathout = 'C:/Users/RuturajMane/Desktop/MNIST_CNN/mnist-ruturaj/Outputs/Test'
   
    pathout = pathout + '/'  + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
    pathout = pathout + '.csv'
    print(pathout)
    return df.to_csv(pathout)

if args.testdataset != None:
    path = args.testdataset
    # assert os.path.isfile(path)
    default(path)

elif args.image == None and args.label == None:
    path = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src\data'
    default(path)

elif args.image != None and args.label != None:
    path = args.image
    assert os.path.isfile(path)
    # with open(path, "r") as f:
    image = Image.open(path)
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(image)
    predict = model(img_tensor)
    precision, recall, fscore, support = score(args.label, predict)

    df = pd.DataFrame(np.array([precision, recall, fscore, support]), index=['precision', 'recall', 'fscore', 'support'])
    df['Average'] = [sum(precision)/len(precision),sum(recall)/len(recall),sum(fscore)/len(fscore),sum(support)/len(support) ]

    if args.outputpath != None:
        pathout = args.outputpath
    
    else:
        pathout = 'C:/Users/RuturajMane/Desktop/MNIST_CNN/mnist-ruturaj/Outputs/Test'
   
    pathout = pathout + '/'  + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
    pathout = pathout + '.csv'
    print(pathout)
    df.to_csv(pathout)






