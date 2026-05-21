import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        for i in range(len(shape)-1):
            self.linear_layers.append(nn.Linear(shape[i], shape[i+1]))
    
    def forward(self, x, return_bitstrings=False):
        bitstrings = []
        x[:, -1] = (x[:, -1] - (500 - 1))/(500 - 1)
        for layer in self.linear_layers[:-1]:
            x = layer(x)
            if return_bitstrings:
                bitstrings.append((x > 0).detach().cpu().numpy())
            x = nn.functional.relu(x)
        x = self.linear_layers[-1](x)
        if return_bitstrings:
            return x, bitstrings
        else:
            return x
        
        
        
# TODO: refactor and update to work with square
def gen_grid_statistics(network, X, Y, time, only_vec_field=False):
    with torch.no_grad():
        xs = X.reshape(-1, 1)
        ys = Y.reshape(-1, 1)
        ts = time*np.ones(xs.shape)
        in_vals = torch.tensor(np.hstack((xs, ys, ts)), dtype=torch.float32)

        out_vals, bitstrings = network(in_vals.to(device), return_bitstrings=True)
        vec_field = out_vals# - in_vals[:, :-1]

        vec_field = vec_field.reshape(X.shape[0], X.shape[1], 2)
        if only_vec_field:
            return -vec_field.detach().cpu().numpy()

        nullclines = vec_field > 0

        grid_bitstrings = []
        for bitstring in bitstrings:
            grid_bitstrings.append(bitstring.reshape(X.shape[0], X.shape[1], bitstring.shape[1]))

        return nullclines.detach().cpu().numpy(), grid_bitstrings
    
def distance_from_circle(points, center, radius):
    dist_from_center = points - center
    point_on_circle = radius*dist_from_center/np.linalg.norm(dist_from_center, axis=1).reshape(-1, 1) + center
    return np.linalg.norm(points - point_on_circle, axis=1)

def distance_from_circles(points):
    circle_loc_1 = np.array([1.5, 1.5])
    circle_r_1 = 0.7
    circle_loc_2 = np.array([-1.5, -1.5])
    circle_r_2 = 0.3
    
    dist_1 = distance_from_circle(points, circle_loc_1, circle_r_1)
    dist_2 = distance_from_circle(points, circle_loc_2, circle_r_2)
    
    return np.minimum(dist_1, dist_2)

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
    def __init__(self, network, colors, alphabar, data_range, contour_bins, vec_bins, true_points, calc_points):
        with torch.no_grad():
            self.network = network
            self.T = len(alphabar)
            self.time = T
            self.data_range = data_range

            self.colors = colors

            x = np.linspace(data_range[0], data_range[1], contour_bins)
            self.X, self.Y = np.meshgrid(x, x)

            x = np.linspace(data_range[0], data_range[1], vec_bins)
            self.X_vec, self.Y_vec = np.meshgrid(x, x)

            self.true_points = dataset.generate_training_data(true_points, alphabar, normalize=True, return_full_only=True)

            normal_distribution = np.random.normal(0, 1, (calc_points, 2))
            in_vals = torch.tensor(np.hstack((normal_distribution, 500*np.ones((calc_points, 1)))), dtype=torch.float32).to(device)
            self.calc_points = in_vals.clone().detach().cpu().numpy()
            for i in range(499, -1, -1):
                #print(self.calc_points)
                vec_field = self.network(in_vals)
                #print(self.calc_points)
                in_vals[:, :-1] =in_vals[:, :-1] - vec_field# torch.hstack((in_vals, i*torch.ones((in_vals.shape[0], 1))))
                in_vals[:, -1] = i
                #print(self.calc_points)
                self.calc_points = np.vstack((self.calc_points, in_vals.detach().cpu().numpy()))
                #print(self.calc_points)
                #break
                
            point_locs_final = self.calc_points[self.calc_points[:, -1] == 0, :-1]
            self.final_MSE = np.mean(distance_from_circles(point_locs_final))

            self.up = in_vals[:, 0].detach().cpu().numpy() > 0
            self.down = in_vals[:, 0].detach().cpu().numpy() <= 0
            torch.cuda.empty_cache()
        
    def calculate(self):
        self.nullclines, self.nonlinearities = gen_grid_statistics(self.network, self.X, self.Y, self.time, only_vec_field=False)
        self.vec_field = gen_grid_statistics(self.network, self.X_vec, self.Y_vec, self.time, only_vec_field=True)
        self.true_points_time = self.true_points[self.true_points[:, -1] == self.time, :-1]
        self.point_locs = self.calc_points[self.calc_points[:, -1] == self.time, :-1]
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
        
        ax_true_point.set_title('Diffused hist')
        ax_true_point.hist2d(self.true_points_time[:, 0], self.true_points_time[:, 1], 
                             bins=100, cmap='plasma', range=[self.data_range, self.data_range])
        
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
        
        times = [500, 300, 100, 0]
        for i in range(4):
            axs[i, 0].set_ylabel(f'T = {times[i]}\n$x_2$')
            
        fig.suptitle(f'Epoch {epochs:>5}', y=0.9)
        
        for i in range(len(times)):
            self.time = times[i]
            self.calculate()
            
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