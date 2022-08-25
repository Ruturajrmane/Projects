import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional
from Dataclasses.hyperpar import Hyperparameters 
from modelparser import ModelParser
from model import ModelParser
from model import MnistModel
import neptune.new as neptune
from utils.common import lssplt, accplt
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Datasetpath', help = 'Enter the dataset path ')
parser.add_argument('--Configpath', help= 'Enter the config path ')

args = parser.parse_args()


# dataset = MNIST(root = 'data/', train = True, download = True)
# test_dataset = MNIST(root = 'data/', train = False)

def preprocess(batch_size, path_data):
    # Converting data into tensor

    dataset = MNIST(root = path_data, train = True, download = True, transform = transforms.ToTensor() )
   
    # Splitting the data into training and validation sets
    train_data, val_data = random_split(dataset, [50000, 10000])

    # Converting data into batches for training and validation
    batch_size = batch_size
    train_loader = DataLoader(train_data,batch_size, shuffle = True)
    # Shuffle set to true to get different batch of data every epoch
    val_loader = DataLoader(val_data, batch_size)
    # In every batch will have image of shape(batchsize, channnel, height, widht) and label of shape(batchsize)

    return train_loader,val_loader

def accuracy(pred, labels):
    _, pred = torch.max(pred).item()
    acc = torch.sum(pred == labels).item()/ len(pred)
    return acc


def fit(epochs, lr, train_loader, val_loader, loss_fn, opt):
    
    # Defining loss function
    if loss_fn in "cross_entropy":
        loss_fnn = torch.nn.functional.cross_entropy

    # if opt in "SGD":
    #     opt = torch.optim.SGD(model.parameters(), lr)
    lt = []
    lv = []
    av = []
    df_f = pd.DataFrame()
    for epoch in range(epochs):
        ltt = []
        l = []
        a = []

        model.train()
        for image, labels in train_loader:
            
            # print(image.shape)
            try:
                pred = model(image)
            except:
                print('Shape of the input to model must be of [No.of inputs of first layer, height of image, width of the image')
            # loss calculation
            loss = loss_fnn(pred, labels) # Passing pred of shape (batchsize, no.of classes) and labels of shape(batchsize)
            # Gradient calculation
            loss.backward()
            # Weight adjusting
            opt.step()
            # Setting gradient to zero
            opt.zero_grad()
            ltt.append(loss)
        
        lt.append((sum(ltt)/ len(ltt)).item())


        labelss = []
        predicts = []  
       
        model.eval() # This disables the dropout and batch normalization as dropout is used for increasing bias (reduce overfitting)
        with torch.no_grad(): # Turns off the gradient calculation false, which makes computation faster. 
            for image, labels in val_loader:
                pred = model(image)
                # loss calculation
                loss = loss_fnn(pred, labels)
                # accuracy calulation
                _, pred = torch.max(pred, dim = 1) # Gives two output first is value and second is index
                acc = (torch.sum(pred == labels).item()/len(pred)) # Pred shape(batchsize) and labels shape(batchsize)

                lab = [t.item() for t in labels]
                for i in lab:
                    labelss.append(i)
                labelss.append(labels)
                pred = [t.item() for t in pred]
                for i in pred:
                    predicts.append(i)
                l.append(loss)
                a.append(acc)
                labelss = labelss[:-1]

        lv.append(sum(l)/len(l))
        av.append(sum(a)/len(a))

        print('Accuracy of epoch {}, is {}'.format((epoch + 1),  sum(a)/len(a)))
        print('Loss of epoch {}, is {}'.format((epoch + 1), sum(l)/len(l) ))
        # run["Val/accu"].log(sum(a)/len(a))
        # run["Val/loss"].log(sum(l)/len(l))

     
        precision, recall, fscore, support = score(labelss, predicts)
        
        dic = {'epochs':epoch,'precision':sum(precision)/len(precision),'recall':sum(recall)/len(recall),'fscore':sum(fscore)/len(fscore),'support':sum(support)/len(support) } 
        
        df = pd.DataFrame(dic, index = [epoch])
      
        df_f = pd.concat([df_f, df], axis = 0)
    # if args.outputpath != None:
    #     pathout = args.outputpath
    
    # else:
    pathout = 'C:/Users/RuturajMane/Desktop/MNIST_CNN/mnist-ruturaj/Logs/Train/'
   
    pathout = pathout + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
    pathout = pathout + '_{0}_{1}_{2}_{3}'.format(batch_size,epochs,lr,loss_fn) + '.csv'
    df_f.to_csv(pathout)
    return lv,av,lt

    
if __name__ == "__main__":

    if args.Configpath == None:
        path_config = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\base_config.json'
    
    else:
        path_config = args.Configpath

    try:
        parser = ModelParser(path_config)
    except:
        print('Couldn\'t import the configartion module')

    layers = parser.get_list()
    kwargs = parser.get_hp()[0]
    model = MnistModel(layers)
    model = model.build_model()
    

    # To do convert to if else
    try:
        assert isinstance(model[0].weight, torch.nn.parameter.Parameter), 'Model isnot initialized'
       
    except AssertionError as msg :
        print(msg)

    Hyperparameters(**kwargs)

    # Assigning hyperparametres
    try:
        batch_size = Hyperparameters.batch_size
        epochs = Hyperparameters.epochs
        lr = Hyperparameters.lr
        opt = Hyperparameters.opt
        if opt == "Adam":
            optm = torch.optim.Adam(model.parameters(), lr)
        elif opt == "SGD":
            optm = torch.optim.SGD(model.parameters(), lr)
        loss_fn = Hyperparameters.loss_fn
    except:
        print('Error in defining the hyperparameters')

    if args.Datasetpath == None:
        path_data = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\src\data'
    
    else:
        path_data = args.Datsetpath

    # Preprocessing
    train_loader,val_loader=preprocess(batch_size, path_data)

    # Neptune model and parametres tracking
    # run = neptune.init(
    # project="ruturaj.mane/MNIST",
    # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZDk2ZGRlZi1kN2MyLTRjNzItOTViMC1mMmNiZGI3NTU5OTAifQ==",
    # )

    # params = {"learning_rate": lr, "Optimizer" : opt, "Batch_size" : batch_size, "Epochs" : epochs, "Loss_fn": loss_fn}
    # run["Parameters"] = params

    lv, av, lt = fit(epochs, lr, train_loader, val_loader, loss_fn, optm)


    lssplt(lv, lt,batch_size,epochs,lr,opt,loss_fn)
    accplt(av,batch_size,epochs,lr,opt,loss_fn)

    # # Log a model as a state_dict
    # with mlflow.start_run():
    #     state_dict = model.state_dict()
    #     mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")

    # run.stop()

    try:
        path = '../Models/MNIST_{0}_{1}_{2}_{3}_{4}'.format(batch_size,epochs,lr,opt,loss_fn)
        path = path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.path'
        torch.save({'model_state': model.state_dict(), 'epochs':epochs, 
        'Learning_rate':lr, 'Loss_fn':loss_fn, 'optmizer_state':optm.state_dict() }, path)

    except:
        print('Path of model saving isnot correct')

    
 
    



    
