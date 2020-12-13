from multiprocessing.spawn import freeze_support
from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from GANs.utility import get_noise
from GANs.utility import get_label_list
from GANs.hyperparamers_commons import *


# Override or add Hyper parameters
img_size = 64  # size of generated image
batch_size = 128
latent = 100  # dim of latent space
img_channel = 1  # channel of generated image
init_channel = 16  # control the initial Conv channel of the Generator and Discriminator
k = 1  # train Discriminator K times and then train Generator one time


def load_data():
    # data enhancement
    data_transform = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # dataset
    data_set = torchvision.datasets.MNIST(
        root='./MNIST',
        download=False,
        train=True,
        transform=data_transform)
    return DataLoader(data_set, batch_size, True, num_workers=workers)

#Created at the start for baseline model. Refactored the code according to newer models implementation, this introduced some compilation errors. Will fix it later

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 128,  4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1, bias=True),
            nn.Tanh()
        )
    def forward(self, inputs):
        return self.main(inputs)


# Discriminator(
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, inputs):
        # x = x.view(-1, 784)
        return self.main(inputs)

def get_noise(noise_num=batch_size):
    return Variable(torch.randn((noise_num, 128)).cuda())

# train the network
def train_model(data_loader):
    start = time()
    fix_noise = get_noise(64)

    # use cuda if you have GPU
    net_g = Generator().cuda()
    print(Generator())
    net_d = Discriminator().cuda()

    # optimizer
    opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))  # optimizer for Generator
    opt_d = torch.optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))  # optimizer for Discriminator
    number = 1
    for epoch in range(epoch_num):
        for step, (real_data, target) in enumerate(data_loader, 1):
            # train Discriminator
            real_data = Variable(real_data).cuda()
            # prob_fake = net_d(net_g(real_data))
            prob_fake = net_d(net_g(get_noise(real_data.size(0))))
            prob_real = net_d(real_data)
            loss_d = - torch.mean(torch.log(prob_real) + torch.log(1 - prob_fake))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # train Generator
            if step % k is 0:
                prob_fake = net_d(net_g(get_noise()))
                loss_g = - torch.mean(torch.log(prob_fake))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            if step % 200 is 0:
                print('epoch:', epoch, 'step', step, 'time:', (time() - start) / 60, 'min', 'generator_loss:',
                      loss_g.item(), "discriminator_loss:", loss_d.item())

                generate_img = torchvision.utils.make_grid((0.5 * net_g(fix_noise).data.cpu() + 0.5))
                generated = generate_img.permute(1, 2, 0).numpy()
                plt.imshow(generated)
                plt.pause(0.01)
                plt.imsave('./generated/dcgan/' + str(number) + '.png', generated)
                number += 1

plt.show()

if __name__ == '__main__':
    freeze_support()
    data_loader = load_data()
    train_model(data_loader)
