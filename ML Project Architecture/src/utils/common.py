import matplotlib.pyplot as pl

def accplt(da):
    pl.plot([(i+1) for i in  range(len(da))], da)
    pl.xlabel('Epochs')
    pl.ylabel('Accuracy')
    pl.title('Accuracy vs Epochs')
    pl.show()
