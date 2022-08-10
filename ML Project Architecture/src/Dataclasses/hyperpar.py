import json
import os

class Hyperparameters:
    def __init__(self,**hyp):
        Hyperparameters.opt = hyp["Optimizer"]
        Hyperparameters.lr = hyp["Learning_rate"]
        Hyperparameters.epochs = hyp["No_of_epochs"]
        Hyperparameters.batch_size = hyp["Batch_size"]
        Hyperparameters.loss_fn = hyp["loss_fun"]

    # @classmethod
    # def prop(cls):
    #     cls.opt = self.hyp["Optimizer"]
    #     cls.lr = self.hyp["Learning_rate"]
    #     cls.epochs = self.hyp["No_of_epochs"]



