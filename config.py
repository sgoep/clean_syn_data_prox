import torch
import numpy as np

class config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    factor = 10
    which = "tv"
    # factor = 10000
    # which = "ell1"
    
    learning_rate = 1e-4
    len_train = 500
    len_test = 100
    epochs = 50
