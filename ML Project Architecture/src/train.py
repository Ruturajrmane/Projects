import torch
import torchvision
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
from utils.common import accplt

# dataset = MNIST(root = 'data/', train = True, download = True)
# test_dataset = MNIST(root = 'data/', train = False)

def preprocess(batch_size):
    # Converting data into tensor
    dataset = MNIST(root = 'data/', train = True, download = True, transform = transforms.ToTensor() )
   
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
        loss_fn = torch.nn.functional.cross_entropy

    # if opt in "SGD":
    #     opt = torch.optim.SGD(model.parameters(), lr)

    dl = []
    da = []

    for epoch in range(epochs):

        l = []
        a = []

        model.train()
        for image, labels in train_loader:
            # print(image.shape)
            pred = model(image)
            # loss calculation
            loss = loss_fn(pred, labels) # Passing pred of shape (batchsize, no.of classes) and labels of shape(btachsize)
            # Gradient calculation
            loss.backward()
            # Weight adjusting
            opt.step()
            # Setting gradient to zero
            opt.zero_grad()
            
        model.eval() # This disables the dropout and batch normalization as dropout is used for increasing bias (reduce overfitting)
        with torch.no_grad(): # Turns off the gradient calculation false, which makes computation faster. 
            for image, labels in val_loader:
                pred = model(image)
                # loss calculation
                loss = loss_fn(pred, labels)
                # accuracy calulation
                _, pred = torch.max(pred, dim = 1) # Gives two output first is value and second is index
                acc = (torch.sum(pred == labels).item()/len(pred)) # Pred shape(batchsize) and labels shape(batchsize)

                l.append(loss)
                a.append(acc)


        dl.append(sum(l)/len(l))
        da.append(sum(a)/len(a))

#         print('Epoch loss : ', torch.stack(l).mean())
#         print('Epoch accu : ', torch.stack(a).mean())
#         print(dl)
        print('Accuracy of epoch {}, is {}'.format((epoch + 1),  sum(a)/len(a)))
        print('Loss of epoch {}, is {}'.format((epoch + 1), sum(l)/len(l) ))
        run["Val/accu"].log(sum(a)/len(a))
        run["Val/loss"].log(sum(l)/len(l))

    return dl,da


if __name__ == "__main__":

    parser = ModelParser("../base_config.json")
    layers = parser.get_list()
    kwargs = parser.get_hp()[0]
    model = MnistModel(layers)
    model = model.build_model()
 
    Hyperparameters(**kwargs)

    # Assigning hyperparametres
    batch_size = Hyperparameters.batch_size
    epochs = Hyperparameters.epochs
    lr = Hyperparameters.lr
    opt = Hyperparameters.opt
    if opt == "Adam":
        optm = torch.optim.Adam(model.parameters(), lr)
    elif opt == "SGD":
        optm = torch.optim.SGD(model.parameters(), lr)
    loss_fn = Hyperparameters.loss_fn

    # Preprocessing
    train_loader,val_loader=preprocess(batch_size)

    # Neptune model and parametres tracking
    run = neptune.init(
    project="ruturaj.mane/MNIST",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZDk2ZGRlZi1kN2MyLTRjNzItOTViMC1mMmNiZGI3NTU5OTAifQ==",
    )

    params = {"learning_rate": lr, "Optimizer" : opt, "Batch_size" : batch_size, "Epochs" : epochs, "Loss_fn": loss_fn}
    run["Parameters"] = params

    dl, da = fit(epochs, lr, train_loader, val_loader, loss_fn, optm)

    accplt(da)
    # # Log a model as a state_dict
    # with mlflow.start_run():
    #     state_dict = model.state_dict()
    #     mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
    
    run.stop()


    torch.save({'model_state': model.state_dict(), 'epochs':epochs, 
    'Learning_rate':lr, 'Loss_fn':loss_fn, 'optmizer_state':optm.state_dict() }, '../Models/MNIST_{0}_{1}_{2}_{3}_{4}.pth'.format(batch_size,epochs,lr,opt,loss_fn))



    
 
    



    
