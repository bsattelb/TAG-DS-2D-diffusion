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
        
    def sample_DDPM(self, npts, device):
        scheduler = self.scheduler
        locations = np.zeros((npts, 2, scheduler.T+1))
        normal_distribution = np.random.normal(0, 1, (npts, 2))
        locations[:, :, -1] = normal_distribution
        with torch.no_grad():
            xs = torch.tensor(normal_distribution, dtype=torch.float32).to(device)
            for t in range(scheduler.T-1, -1, -1):
                if t > 1:
                    z = torch.randn_like(xs)
                else:
                    z = torch.zeros_like(xs)

                e_theta = self.forward(xs, t)
                update = (1 - scheduler.alphas[t])/np.sqrt(1 - scheduler.alphabars[t])*e_theta
                xs = (xs - update)/np.sqrt(scheduler.alphas[t]) + scheduler.sigmas[t]*z

                locations[:, :, t] = xs.detach().cpu().numpy()
        return locations
            
            