import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import sqrt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, epoch, type, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    plt.savefig("conv2d_results/"+type+"conv2d"+str(epoch)+".png")

def get_noise(n_samples, z_dim, device='cuda'):
    return torch.reshape(torch.randn(n_samples, z_dim,device=device), (n_samples, 1, int(sqrt(z_dim)), int(sqrt(z_dim))))

class Generator(nn.Module):
    def __init__(self, channles=1, d=4):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(channles, d*2, 3, 3),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
            nn.ConvTranspose2d(d*2, d, 3, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.ConvTranspose2d(d, channles, 3, 1),
            nn.BatchNorm2d(channles),
            nn.Sigmoid()
        )
        
    def forward(self, noise):
        return self.gen(noise)
    
    def get_gen(self):        
        return self.gen

class Discriminator(nn.Module):
    def __init__(self, channles=1, d=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, d, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(d, 1, 3, 1),
            nn.Flatten(),
            nn.Linear(26*26, 1),
        )

    def forward(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.disc

criterion = nn.BCEWithLogitsLoss()
n_epochs = 11
z_dim = 64
channles = 1
batch_size = 512
lr = 0.00001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

gen = Generator().to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):    
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise.detach())
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss
  
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

for epoch in range(n_epochs):    
    mean_generator_loss = []
    mean_discriminator_loss = []
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        real = real.to(device)
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()
        mean_discriminator_loss.append(disc_loss.item())
        mean_generator_loss.append(gen_loss.item())

    print(f"Epoch {epoch}: Generator loss: {sum(mean_generator_loss)/len(mean_generator_loss)}, discriminator loss: {sum(mean_discriminator_loss)/len(mean_discriminator_loss)}")
    if epoch%5 ==0:
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        show_tensor_images(fake, epoch, "fake")
        show_tensor_images(real, epoch, "real")