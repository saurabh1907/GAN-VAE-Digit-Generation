from multiprocessing.spawn import freeze_support
from time import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from GANs.utility import get_noise
from GANs.utility import get_label_list

# Override or add Hyper parameters
from GANs.hyperparamers_commons import *
img_size = 64  # size of generated image
batch_size = 128
latent = 100  # dim of latent space
conditional_code = 10  # dim of conditional code
img_channel = 1  # channel of generated image
init_channel = 16  # control the initial Conv channel of the Generator and Discriminator
k = 2  # train Discriminator K times and then train Generator one time

def load_data():
    # data enhancement
    data_transform = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # dataset
    data_set = torchvision.datasets.MNIST('./MNIST/', transform=data_transform)
    data_loader = DataLoader(data_set, batch_size, True, num_workers=workers)
    return data_loader


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1_1 = nn.Sequential(  # input for latent coda
            nn.ConvTranspose2d(latent, init_channel * 4, 4, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU(),
        )

        self.deconv1_2 = nn.Sequential(  # input for conditional code
            nn.ConvTranspose2d(conditional_code, init_channel * 4, 4, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 8, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 4, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.ReLU(),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(init_channel * 2, init_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(init_channel, img_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # initialization for parameters

        for layer in self.modules():

            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, inputs, labels):

        outputs = torch.cat((self.deconv1_1(inputs), self.deconv1_2(labels)), dim=1)

        outputs = self.deconv2(outputs)
        outputs = self.deconv3(outputs)
        outputs = self.deconv4(outputs)
        outputs = self.deconv5(outputs)

        return outputs


# Discriminator(
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Sequential(  # input for image
            nn.Conv2d(img_channel, init_channel // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv1_2 = nn.Sequential(  # input for onehot code
            nn.Conv2d(conditional_code, init_channel // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(slope)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channel, init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 2),
            nn.LeakyReLU(slope)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(init_channel * 2, init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 4),
            nn.LeakyReLU(slope)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(init_channel * 4, init_channel * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_channel * 8),
            nn.LeakyReLU(slope)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(init_channel * 8, 1, 4, bias=False),
            nn.Sigmoid()
        )

        # initialization for parameters
        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, inputs, labels):

        outputs = torch.cat((self.conv1_1(inputs), self.conv1_2(labels)), dim=1)

        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)

        return outputs.view(inputs.size(0))

# transform a number to one hot code
def one_hot(number):

    temp = torch.zeros((number.size(0), conditional_code))

    for i in range(number.size(0)):

        temp[i][int(number[i])] = 1

    temp = temp.view(temp.size(0), temp.size(1), 1, 1)

    return Variable(temp).cuda()


# expand a onehot code to image size
def one_hot_expand(onehot_code):
    return onehot_code.expand(onehot_code.size(0), onehot_code.size(1), img_size, img_size)



# train the network
def train_model(data_loader):
    start = time()
    labellist = get_label_list()
    # use cuda if you have GPU
    net_g = Generator().cuda()
    net_d = Discriminator().cuda()

    # optimizer
    opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))  # optimizer for Generator
    opt_d = torch.optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))  # optimizer for Discriminator

    number = 1
    for epoch in range(epoch_num):
        for step, (real_data, target) in enumerate(data_loader, 1):
            # train Discriminator
            real_data = Variable(real_data).cuda()
            real_label = one_hot(target)
            fake_label = one_hot(torch.floor(torch.rand(real_data.size(0)) * conditional_code))
            prob_fake = net_d(net_g(get_noise(real_data.size(0)), fake_label), one_hot_expand(fake_label))
            prob_real = net_d(real_data, one_hot_expand(real_label))
            loss_d = - torch.mean(torch.log(prob_real) + torch.log(1 - prob_fake))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # train Generator
            if step % k is 0:
                fake_label = one_hot(torch.floor(torch.rand(batch_size) * conditional_code))
                prob_fake = net_d(net_g(get_noise(), fake_label), one_hot_expand(fake_label))
                loss_g = - torch.mean(torch.log(prob_fake))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            if step % 200 is 0:
                # print('epoch:', epoch, 'step', step, 'time:', (time() - start) / 60, 'min')
                print('epoch:', epoch, 'step', step, 'time:', (time() - start) / 60, 'min', 'generator_loss:', loss_g.item(), "discriminator_loss:", loss_d.item())
                fake_label = one_hot(torch.FloatTensor(labellist))
                generate_img = torchvision.utils.make_grid((0.5 * net_g(get_noise(80), fake_label).data.cpu() + 0.5))
                generated = generate_img.permute(1, 2, 0).numpy()
                plt.imshow(generated)
                plt.pause(0.01)
                plt.imsave('./generated/cgan/' + str(number) + '.png', generated)
                number += 1

if __name__ == '__main__':
    freeze_support()
    data_loader = load_data()
    train_model(data_loader)
