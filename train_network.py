import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import os

import dataset
import network
import nonlinearity

class Scheduler:
    def __init__(self, T):
        self.T=T
        self.betas = np.linspace(1e-4, 0.01, T)
        
        self.alphas = 1 - self.betas
        # Note that alpha in the DDIM paper refers to alphabar
        self.alphabars = np.cumprod(self.alphas)
        
        alphabar_t_1 = np.insert(self.alphabars[:-1], 0, 1)
        beta_bar = (1 - alphabar_t_1)/(1 - self.alphabars)*self.betas
        self.sigmas = np.sqrt(beta_bar)
        


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Diffusion parameters
    # t = 0 - circles
    # t = 1 - first addition of noise
    # t = 500 - all noise added, ostensibly normally distributed
    T = 1000
    scheduler = Scheduler(T)

    # Training parameters
    criterion = nn.MSELoss(reduction='mean')
    npts = 1000
    OoMs = 6
    opt = lambda params: torch.optim.Adam(params)
    network_shape = [3, 100, 100, 100, 2]
    batch_size = 64

    # Plotting parameters
    grid_points=400
    vec_grid_points=20
    hist_points = 100000
    num_colors = max(network_shape)
    colors = []
    for i in range(num_colors):
        colors.append((np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)))
    data_range = [-3, 3]

    # Saving_parameters
    checkpoint_path = 'checkpoints/'
    animation_path = 'animation_e{}'
    epochs_path = 'animation_epoch'
    if not os.path.isdir(epochs_path + '_DDPM'):
        os.mkdir(epochs_path + '_DDPM')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
        
    net = network.Network(network_shape, scheduler).to(device)
    optimizer = opt(net.parameters())
    
    losses = []
    samples = []
    samples_seen = 0
    alph = torch.Tensor(scheduler.alphabars).to(device)
    Ts = list(range(1, T))
    for i in tqdm(range(OoMs)):
        if i > 0:
            npts *= 10

        lim = 10 if i == 0 else 9
        for j in tqdm(range(lim), leave=False):
            inputs = dataset.generate_initial_datapoints(npts, normalize=True).to(device)
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs), 
                                                       batch_size=batch_size, shuffle=True)
            full_loss = 0
            for inputs in tqdm(train_loader, leave=False, total=len(train_loader)):
                inputs = inputs[0]
                t = torch.randint(0, T, (inputs.shape[0],1))
                e = torch.randn_like(inputs, requires_grad=False)
                a = alph[t]
                x = torch.sqrt(a)*inputs + torch.sqrt(1-a)*e
                output = net(x, t.to(device))
                optimizer.zero_grad()
                loss = criterion(output, e)
                full_loss += loss.item()
                loss.backward()
                optimizer.step()

            samples_seen += len(train_loader.dataset)
            losses.append(full_loss/npts)
            samples.append(len(train_loader.dataset))

            temp = nonlinearity.Animator(net, colors, scheduler, data_range, 
                                         grid_points, vec_grid_points, 
                                         hist_points, device, is_DDPM=True)
            temp.draw_small(samples_seen, times=[999, 333, 100, 0], folder=epochs_path + '_DDPM')
            plt.close()

            checkpoint = {
                    'weights': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'ema': ema.state_dict()
            }
            np.save(checkpoint_path + '/losses.npy', losses)
            np.save(checkpoint_path + '/samples.npy', samples)

            torch.save(checkpoint, checkpoint_path + 'nonlinears_e{}.pt'.format(3+i))