import numpy as np
import torch

def circle_outline(center, radius, num_points):
    t = np.random.uniform(0, 1, num_points)
    x = radius*np.cos(2*np.pi*t) + center[0]
    y = radius*np.sin(2*np.pi*t) + center[1]
    
    return np.vstack([x, y]).T

def square_fill(left, down, right, up, num_points):
    x = np.random.uniform(left, right, num_points)
    y = np.random.uniform(up, down, num_points)
    
    return np.vstack([x, y]).T

def generate_datapoints(circ_center, circ_radius, square_left, square_down, 
                  square_right, square_up, square_proportion, num_points):
    
    data = np.zeros((num_points, 2))
    squares = np.random.uniform(0, 1, num_points) < square_proportion
    square_count = np.sum(squares)
    
    data[squares, :] = square_fill(square_left, square_down, square_right, square_up, square_count)
    data[np.logical_not(squares), :] = circle_outline(circ_center, circ_radius, num_points - square_count)
    
    return data

def generate_initial_datapoints(npts_t0, circ_center=[1, 1], circ_radius=0.5, 
                           square_left=-1, square_down=-1, square_right=0, 
                           square_up=0, square_proportion=0.6, normalize=False):
  
    x0s = generate_datapoints(circ_center, circ_radius, square_left, square_down,
                              square_right, square_up, square_proportion, npts_t0)
    
    if normalize:
        x0s = x0s - np.mean(x0s, axis=0)
        x0s = x0s/np.std(x0s, axis=0)
    
    return torch.tensor(x0s, dtype=torch.float32)