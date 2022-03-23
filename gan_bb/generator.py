import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from swin import SwinTransformer3D
import argparse
from args import gan_args
from collections import OrderedDict

parser = argparse.ArgumentParser()
gan_args(parser)
args = parser.parse_args()

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


batch_size = 128
image_size = 64 # Spatial size of training images. All images will be resized to this size using a transformer.
nc = 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in generator
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x

class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples, video_len=None):
        z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z = torch.cat([z_content, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(num_samples, video_len)

        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, Variable(z_category_labels, requires_grad=False)

    def sample_images(self, num_samples):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())


class Disciminator(nn.Module):
    def __init__(self, swint_path) -> None:
        super().__init__()
        self.model = SwinTransformer3D(
                          embed_dim=96, 
                          depths=[2, 2, 6, 2], 
                          num_heads=[3, 6, 12, 24], 
                          patch_size=(2,4,4), 
                          window_size=(8,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True
                          )
        checkpoint = torch.load(swin_wt_path)
        
        new_state_dict = OrderedDict()
    
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
                
        self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        """
        :param x: shape (bs, channels, time, H, W)
        """
        out = self.model(x)
        return out

class Correlator(nn.Module):
    def __init__(self, num_labels) -> None:
        super().__init__()
        self.l1 = nn.Linear(num_labels, 128)
        self.l2 = nn.Linear(num_labels, 64)
        self.l3 = nn.Linear(num_labels, 128)
        self.l4 = nn.Linear(128, num_labels)
        self.relu = nn.ReLU(-1)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.softmax(self.l4(x))
        return x 

swin_wt_path = "/home/abhijeet/bosch/swin_tiny_patch244_window877_kinetics400_1k.pth"
loss_fn = nn.CrossEntropyLoss()

def main(args, training_class):
    netG = VideoGenerator(args.n_channels, args.dim_z_content, args.dim_z_category, args.dim_z_motion, args.video_length)
    netD = Disciminator(swin_wt_path)
    netC = Correlator(args.num_labels)
    true_labels = torch.ones(args.bs).to(device)*training_class
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        netC.cuda()

    opt_netG = optim.Adam(netG.parameters(), lr = args.lr_g)
    opt_netC = optim.Adam(netC.parameters(), lr = args.lr_c)
    
    for i in range(args.num_epochs):
        opt_netG.zero_grad()
        opt_netC.zero_grad()

        videos, _ = netG.sample_videos(args.bs)
        with torch.no_grad():
            out = torch.argmax(netC(videos))
        
        onehot = torch.zeros((args.bs, args.num_labels)).to(device)
        onehot[torch.arange(args.bs), out] = 1

        final_out = netC(onehot)
        loss = loss_fn(final_out, true_labels)
        loss.backward()
        opt_netC.step()
        opt_netG.step()
        print(loss)

main(args, 0)