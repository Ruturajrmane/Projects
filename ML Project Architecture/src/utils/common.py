import matplotlib.pyplot as pl
from datetime import datetime

def lssplt(lv,lt,batch_size,epochs,lr,opt,loss_fn):
    pl.plot([(i+1) for i in  range(len(lv))], lv, label = 'Validation Loss' )
    pl.plot([(i+1) for i in  range(len(lt))], lt, label= 'Training Loss')
    pl.xlabel('Epochs')
    pl.ylabel('Loss')
    pl.title('Loss vs Epochs')
    pl.legend()
    path = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\Outputs\Train\Loss_vs_Epochs' 
    path = path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '_{0}_{1}_{2}_{3}_{4}'.format(batch_size,epochs,lr,opt,loss_fn) + '.png'
    pl.savefig(path)
    pl.show()

def accplt(av,batch_size,epochs,lr,opt,loss_fn):
    pl.plot([(i+1) for i in  range(len(av))], av)
    pl.xlabel('Epochs')
    pl.ylabel('Accuracy')
    pl.title('Accuracy vs Epochs of Validation')
    path = r'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\Outputs\Train\Accuracy_vs_Epochs_of_Validation' 
    path = path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '_{0}_{1}_{2}_{3}_{4}'.format(batch_size,epochs,lr,opt,loss_fn) +'.png'
    #path = 'C:\Users\RuturajMane\Desktop\MNIST_CNN\mnist-ruturaj\Outputs\Train\Accuracy vs Epochs of Validation' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.png'
    pl.savefig(path)
    pl.show()
    
