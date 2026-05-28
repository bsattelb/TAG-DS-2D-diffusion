import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self, shape, scheduler):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        for i in range(len(shape)-1):
            self.linear_layers.append(nn.Linear(shape[i], shape[i+1]))
        self.scheduler = scheduler
    
    def forward(self, x, t, return_bitstrings=False):
        if type(t) == int:
            t = t*torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        t = (t - (self.scheduler.T - 1))/(self.scheduler.T - 1)
        
        x = torch.cat((x, t), dim=1)
        bitstrings = []
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
        
    # sigma_scale = 0 yields DDIM
    # sigma_scale = 1 yields DDPM
    def reverse_diffuse(self, x, t, sigma_scale, return_bitstrings=False, return_vec_field=False):
        e_theta, bitstrings = self.forward(x, t, return_bitstrings=True)
        
        if t > 0:
            z = torch.randn_like(x)
            alph_t_1 = self.scheduler.alphabars[t-1]
        else:
            z = torch.zeros_like(x)
            alph_t_1 = 1
            
        noise = sigma_scale*self.scheduler.sigmas[t]*z
        
        vec_field = -np.sqrt(1 - self.scheduler.alphabars[t])*e_theta
        vec_field += np.sqrt(self.scheduler.alphas[t])*np.sqrt(1 - alph_t_1 - self.scheduler.sigmas[t]**2)*e_theta
            
        x_new = (x + vec_field)/np.sqrt(self.scheduler.alphas[t]) + noise
        
        if return_bitstrings and return_vec_field:
            return x_new, bitstrings, vec_field
        if return_bitstrings:
            return x_new, bitstrings
        if return_vec_field:
            return x_new, vec_field
        return x_new
        
    def sample(self, npts, sigma_scale, device):
        locations = np.zeros((npts, 2, self.scheduler.T+1))
        normal_distribution = np.random.normal(0, 1, (npts, 2))
        locations[:, :, -1] = normal_distribution
        
        with torch.no_grad():
            xs = torch.tensor(normal_distribution, dtype=torch.float32).to(device)
            for t in range(self.scheduler.T-1, -1, -1):
                xs = self.reverse_diffuse(xs, t, sigma_scale)
                locations[:, :, t] = xs.detach().cpu().numpy()
                
        return locations
            
            