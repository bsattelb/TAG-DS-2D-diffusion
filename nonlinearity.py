import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches
import os
from tqdm.notebook import tqdm

def plot_nonlinearities(X, Y, nonlinearities, colors,
                       ax_l1_nonlinear,
                       ax_l2_nonlinear,
                       ax_l3_nonlinear):
    ax_l1_nonlinear.set_title('Layer one nonlinearities')
    ax_l2_nonlinear.set_title('Layer two nonlinearities')
    ax_l3_nonlinear.set_title('Layer three nonlinearities')
    for i in range(nonlinearities[0].shape[2]):
        ax_l1_nonlinear.contour(X, Y, nonlinearities[0][:, :, i], levels=0, linewidths=1, colors=colors[i], alpha=0.5)
    for i in range(nonlinearities[1].shape[2]):
        ax_l2_nonlinear.contour(X, Y, nonlinearities[1][:, :, i], levels=0, linewidths=1, colors=colors[i], alpha=0.5)
    for i in range(nonlinearities[2].shape[2]):
        ax_l3_nonlinear.contour(X, Y, nonlinearities[2][:, :, i], levels=0, linewidths=1, colors=colors[i], alpha=0.5)
        
def plot_nullclines(X, Y, nullclines, ax_nullclines):
    ax_nullclines.contour(X, Y, nullclines[:, :, 0], levels=0, linewidths=1, colors='r', alpha=0.5)
    ax_nullclines.contour(X, Y, nullclines[:, :, 1], levels=0, linewidths=1, colors='b', alpha=0.5)
    red = mpatches.Patch(color='red', label='$\Delta x_1 = 0$')
    blue = mpatches.Patch(color='blue', label='$\Delta x_2 = 0$')
    ax_nullclines.legend(handles=[red, blue])

class Animator:
    def __init__(self, network, colors, scheduler, data_range, contour_bins, vec_bins,
                 calc_points, device='cpu', is_DDPM=True, points_in_cloud=1000):
        self.network = network
        self.scheduler = scheduler
        self.data_range = data_range
        self.device = device
        self.points_in_cloud = points_in_cloud
        
        self.is_DDPM = is_DDPM
        if is_DDPM:
            self.sigma_scale = 1
        else:
            self.sigma_scale = 0 # DDIM

        self.colors = colors

        # Grid for approximating nonlinearities
        x = np.linspace(data_range[0], data_range[1], contour_bins)
        self.X, self.Y = np.meshgrid(x, x)

        # Stride for calculating vector field
        self.vec_stride = contour_bins // vec_bins
        if self.vec_stride < 1:
            self.vec_stride = 1
        #x = np.linspace(data_range[0], data_range[1], vec_bins)
        #self.X_vec, self.Y_vec = np.meshgrid(x, x)

        self.calc_points = network.sample(calc_points, self.sigma_scale, device)
        self.up = self.calc_points[:, 0, 0] > 0
        self.down = self.calc_points[:, 1, 0] < 0
        
        self.calculate(scheduler.T-1)
        
    def gen_nonlinear_structure(self):
        t = self.time
        with torch.no_grad():
            xs = self.X.reshape(-1, 1)
            ys = self.Y.reshape(-1, 1)
            
            in_vals = torch.tensor(np.hstack((xs, ys)), dtype=torch.float32).to(self.device)
            _, bitstrings, vec_field = self.network.reverse_diffuse(in_vals, t, 
                                                                   self.sigma_scale,
                                                                   return_bitstrings=True, 
                                                                   return_vec_field=True)

            vec_field = vec_field.detach().cpu().numpy()
            vec_field = vec_field.reshape(self.X.shape[0], self.X.shape[1], 2)
            nullclines = vec_field > 0
            
            grid_bitstrings = []
            for bitstring in bitstrings:
                grid_bitstrings.append(bitstring.reshape(self.X.shape[0], self.X.shape[1], bitstring.shape[1]))
                
            self.nullclines = nullclines
            self.nonlinearities = grid_bitstrings
            
            start = self.vec_stride//2
            self.X_vec = self.X[start::self.vec_stride, start::self.vec_stride]
            self.Y_vec = self.Y[start::self.vec_stride, start::self.vec_stride]
            self.vec_field = vec_field[start::self.vec_stride, start::self.vec_stride]
        
    def calculate(self, t):
        self.time = t
        self.gen_nonlinear_structure()
        self.point_locs = self.calc_points[:, :, self.time]
        torch.cuda.empty_cache()
        
    def draw(self, folder=None):
        ax_l1_nonlinear = self.fig.add_subplot(231)
        ax_l2_nonlinear = self.fig.add_subplot(232)
        ax_l3_nonlinear = self.fig.add_subplot(233)
        ax_prob_dens_full = self.fig.add_subplot(234)
        #ax_prob_dens_up = self.fig.add_subplot(335)
        #ax_prob_dens_down = self.fig.add_subplot(336)
        #ax_true_point = self.fig.add_subplot(337)
        ax_point_cloud = self.fig.add_subplot(235)
        ax_vec_field = self.fig.add_subplot(236)
        
        for ax in self.fig.axes:
            ax.set_ylim([-3, 3])
            ax.set_xlim([-3, 3])
            ax.set_aspect('equal')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            
        self.fig.suptitle(f'Time {self.time+1: >4}$ \\to ${self.time: >4}')
        
        plot_nonlinearities(self.X, self.Y, self.nonlinearities, self.colors, ax_l1_nonlinear, ax_l2_nonlinear, ax_l3_nonlinear)
        
        ax_prob_dens_full.set_title('Histogram')
        ax_prob_dens_full.hist2d(self.point_locs[:, 0], self.point_locs[:, 1], 
                                 bins=100, cmap='plasma', range=[self.data_range, self.data_range])
        
        ax_point_cloud.set_title('Point cloud')
        temp = self.point_locs[:self.points_in_cloud, :]
        up = self.up[:self.points_in_cloud]
        down = self.down[:self.points_in_cloud]
        ax_point_cloud.scatter(temp[up, 0], temp[up, 1], s=1, c='y', label='Circle')
        ax_point_cloud.scatter(temp[down, 0], temp[down, 1], s=1, c='m', label='Square')
        ax_point_cloud.legend()
        
        if self.is_DDPM:
            ax_vec_field.set_title('DDPM Vector Field')
        else:
            ax_vec_field.set_title('DDIM Vector Field')
        ax_vec_field.quiver(self.X_vec, self.Y_vec, self.vec_field[:, :, 0], self.vec_field[:, :, 1])
        plot_nullclines(self.X, self.Y, self.nullclines, ax_vec_field)
        
        if folder:
            self.fig.savefig(f'{folder}/{self.scheduler.T-self.time:04d}', bbox_inches='tight')
            plt.clf()
        
    def make_animation(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.fig = plt.figure(figsize=(15, 10), layout='constrained')
        for i in tqdm(range(self.scheduler.T), leave=False):
            self.calculate(i)
            self.draw(folder=folder)
            
    def draw_small(self, epochs, times=None, folder=None):
        fig = plt.figure(figsize=(25, 20))
        
        if times is None:
            times = [self.scheduler.T-1, (self.scheduler.T*2)//3, self.scheduler.T//3, 0]
            
        axs = fig.subplots(len(times), 5)
        for ax in fig.axes:
            ax.set_ylim([-3, 3])
            ax.set_xlim([-3, 3])
            ax.set_aspect('equal')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
        for i in range(4):
            axs[i, 0].set_ylabel(f'Time ${times[i]+1}\\to{times[i]}$\n$x_2$')
            
        fig.suptitle(f'{epochs:.0E} samples seen', fontsize=24)
        
        for i in range(len(times)):
            self.calculate(times[i])
            
            plot_nonlinearities(self.X, self.Y, self.nonlinearities, self.colors, axs[i, 0], axs[i, 1], axs[i, 2])
            
            if self.is_DDPM:
                axs[i, 3].set_title('DDPM Vector Field')
            else:
                axs[i, 3].set_title('DDIM Vector Field')
            axs[i, 3].quiver(self.X_vec, self.Y_vec, self.vec_field[:, :, 0], self.vec_field[:, :, 1])
            plot_nullclines(self.X, self.Y, self.nullclines, axs[i, 3])
            
            axs[i,4].set_title('Point cloud')
            temp = self.point_locs[:self.points_in_cloud, :]
            up = self.up[:self.points_in_cloud]
            down = self.down[:self.points_in_cloud]
            axs[i,4].scatter(temp[up, 0], temp[up, 1], s=1, c='y', label='Circle')
            axs[i,4].scatter(temp[down, 0], temp[down, 1], s=1, c='m', label='Square')
            axs[i,4].legend()
            
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if folder:
            fig.savefig(f'{folder}/{epochs:011d}', bbox_inches='tight')