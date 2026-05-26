import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches

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
    ax_nullclines.set_title('Nullclines')
    ax_nullclines.contour(X, Y, nullclines[:, :, 0], levels=0, linewidths=1, colors='r', alpha=0.5)
    ax_nullclines.contour(X, Y, nullclines[:, :, 1], levels=0, linewidths=1, colors='b', alpha=0.5)
    red = mpatches.Patch(color='red', label='$\Delta x_1 = 0$')
    blue = mpatches.Patch(color='blue', label='$\Delta x_2 = 0$')
    ax_nullclines.legend(handles=[red, blue])

class Animator:
    def __init__(self, network, colors, scheduler, data_range, contour_bins, vec_bins,
                 true_points, calc_points, device='cpu'):
        self.network = network
        self.scheduler = scheduler
        self.data_range = data_range
        self.device = device

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

        self.calc_points = network.sample_DDPM(calc_points, device)
        self.up = self.calc_points[:, 0, 0] > 0
        self.down = self.calc_points[:, 1, 0] > 0
        
        self.calculate(scheduler.T-1)
        
    def gen_nonlinear_structure(self):
        t = self.time
        with torch.no_grad():
            xs = self.X.reshape(-1, 1)
            ys = self.Y.reshape(-1, 1)
            
            in_vals = torch.tensor(np.hstack((xs, ys)), dtype=torch.float32).to(self.device)
            _, bitstrings, vec_field = self.network.reverse_diffuse_DDPM(in_vals, t, 
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
            
            self.X_vec = self.X[::self.vec_stride, ::self.vec_stride]
            self.Y_vec = self.Y[::self.vec_stride, ::self.vec_stride]
            self.vec_field = vec_field[::self.vec_stride, ::self.vec_stride]
        
#     def gen_vec_field(self):
#         t = self.time
#         with torch.no_grad():
#             xs = self.X_vec.reshape(-1, 1)
#             ys = self.Y_vec.reshape(-1, 1)
#             in_vals = torch.tensor(np.hstack((xs, ys)), dtype=torch.float32)

#             out_vals = self.network(in_vals.to(self.device), t)
#             field = -(1 - self.scheduler.alphas[t])/np.sqrt(1 - self.scheduler.alphabars[t])*out_vals

#             field = field.reshape(self.X_vec.shape[0], self.X_vec.shape[1], 2)
#             self.vec_field = field.detach().cpu().numpy()
    
#     def gen_nonlinearities(self):
#         t = self.time
#         with torch.no_grad():
#             xs = self.X.reshape(-1, 1)
#             ys = self.Y.reshape(-1, 1)
#             in_vals = torch.tensor(np.hstack((xs, ys)), dtype=torch.float32)

#             out_vals, bitstrings = self.network(in_vals.to(self.device), t, return_bitstrings=True)
#             out_vals = out_vals.reshape(self.X.shape[0], self.X.shape[1], 2)
#             nullclines = (1 - self.scheduler.alphas[t])/np.sqrt(1 - self.scheduler.alphabars[t])*out_vals > 0

#             grid_bitstrings = []
#             for bitstring in bitstrings:
#                 grid_bitstrings.append(bitstring.reshape(self.X.shape[0], self.X.shape[1], bitstring.shape[1]))

#         self.nullclines = nullclines.detach().cpu().numpy()
#         self.nonlinearities = grid_bitstrings
        
    def calculate(self, t):
        self.time = t
        self.gen_nonlinear_structure()
        self.point_locs = self.calc_points[:, :, self.time]
        torch.cuda.empty_cache()
        
    def draw(self):
        ax_l1_nonlinear = self.fig.add_subplot(331)
        ax_l2_nonlinear = self.fig.add_subplot(332)
        ax_l3_nonlinear = self.fig.add_subplot(333)
        ax_prob_dens_full = self.fig.add_subplot(334)
        ax_prob_dens_up = self.fig.add_subplot(335)
        ax_prob_dens_down = self.fig.add_subplot(336)
        ax_true_point = self.fig.add_subplot(337)
        ax_point_cloud = self.fig.add_subplot(338)
        ax_vec_field = self.fig.add_subplot(339)
        
        for ax in self.fig.axes:
            ax.set_ylim([-3, 3])
            ax.set_xlim([-3, 3])
            ax.set_aspect('equal')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            
        self.fig.suptitle(f'T = {self.time:>3}')
        
        plot_nonlinearities(self.X, self.Y, self.nonlinearities, self.colors, ax_l1_nonlinear, ax_l2_nonlinear, ax_l3_nonlinear)
        
        ax_prob_dens_full.set_title('Histogram')
        ax_prob_dens_full.hist2d(self.point_locs[:, 0], self.point_locs[:, 1], 
                                 bins=100, cmap='plasma', range=[self.data_range, self.data_range])
        
        ax_prob_dens_up.set_title('Upper circle hist')
        ax_prob_dens_up.hist2d(self.point_locs[self.up, 0], self.point_locs[self.up, 1], 
                               bins=100, cmap='plasma', range=[self.data_range, self.data_range])
    
        ax_prob_dens_down.set_title('Lower circle hist')
        ax_prob_dens_down.hist2d(self.point_locs[self.down, 0], self.point_locs[self.down, 1], 
                                 bins=100, cmap='plasma', range=[self.data_range, self.data_range])
        
#         ax_true_point.set_title('Diffused hist')
#         ax_true_point.hist2d(self.true_points_time[:, 0], self.true_points_time[:, 1], 
#                              bins=100, cmap='plasma', range=[self.data_range, self.data_range])
        
        ax_point_cloud.set_title('Point cloud')
        temp = self.point_locs[:200, :]
        up = self.up[:200]
        down = self.down[:200]
        ax_point_cloud.scatter(temp[up, 0], temp[up, 1], s=1, c='y', label='TR')
        ax_point_cloud.scatter(temp[down, 0], temp[down, 1], s=1, c='m', label='LL')
        ax_point_cloud.legend()
        
        ax_vec_field.set_title('Vector Field')
        ax_vec_field.quiver(self.X_vec, self.Y_vec, self.vec_field[:, :, 0], self.vec_field[:, :, 1])
        plot_nullclines(self.X, self.Y, self.nullclines, ax_vec_field)
        
        self.fig.savefig(f'{self.folder}/{self.T-self.time:03d}', bbox_inches='tight')
        plt.clf()
        
    def runner(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.folder = folder

        self.fig = plt.figure(figsize=(15, 15), layout='constrained')
        for i in tqdm(range(501), leave=False):
            self.time = i
            self.calculate()
            self.draw()
            
    def draw_small(self, epochs, folder=None):
        fig = plt.figure(figsize=(25, 20))
        axs = fig.subplots(4, 5)
        for ax in fig.axes:
            ax.set_ylim([-3, 3])
            ax.set_xlim([-3, 3])
            ax.set_aspect('equal')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
        
        # TODO: fix off by one error
        times = [499, 300, 100, 0]
        for i in range(4):
            axs[i, 0].set_ylabel(f'T = {times[i]}\n$x_2$')
            
        fig.suptitle(f'Epoch {epochs:>5}', y=0.9)
        
        for i in range(len(times)):
            self.calculate(times[i])
            
            plot_nonlinearities(self.X, self.Y, self.nonlinearities, self.colors, axs[i, 0], axs[i, 1], axs[i, 2])
            
            axs[i, 3].set_title('Vector Field')
            axs[i, 3].quiver(self.X_vec, self.Y_vec, self.vec_field[:, :, 0], self.vec_field[:, :, 1])
            plot_nullclines(self.X, self.Y, self.nullclines, axs[i, 3])
            
            axs[i,4].set_title('Point cloud')
            temp = self.point_locs[:200, :]
            up = self.up[:200]
            down = self.down[:200]
            axs[i,4].scatter(temp[up, 0], temp[up, 1], s=1, c='y', label='TR')
            axs[i,4].scatter(temp[down, 0], temp[down, 1], s=1, c='m', label='LL')
            axs[i,4].legend()
            
        if folder:
            fig.savefig(f'{folder}/{epochs:04d}', bbox_inches='tight')
            plt.clf()