import torch
from torch.autograd import Variable
# get random noise
batch_size = 128
latent = 100

def get_noise(noise_num=batch_size):
    return Variable(torch.randn((noise_num, latent, 1, 1)).cuda())

def get_label_list():
    labellist = []
    for i in range(10):
        for j in range(8):
            labellist.append(i)
    return labellist