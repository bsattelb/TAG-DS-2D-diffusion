import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ripser import ripser

import network
from train_network import Scheduler

# -------------------
# Setup
# -------------------
device = "cpu"

T = 1000
scheduler = Scheduler(T)

net = network.Network([3, 100, 100, 100, 2], scheduler).to(device)

checkpoint = torch.load(
    "network_final.pt",
    map_location=device
)
net.load_state_dict(checkpoint["weights"])
net.eval()

# scheduler tensors
betas = torch.tensor(scheduler.betas, dtype=torch.float32, device=device)
alphas = torch.tensor(scheduler.alphas, dtype=torch.float32, device=device)
alphabars = torch.tensor(scheduler.alphabars, dtype=torch.float32, device=device)

# -------------------
# Sample reverse diffusion trajectory
# -------------------
npts = 300
locations = np.zeros((npts, 2, T + 1))

# t = T is pure noise
x_t = np.random.normal(0, 1, (npts, 2))
locations[:, :, T] = x_t

with torch.no_grad():
    x_t = torch.tensor(x_t, dtype=torch.float32, device=device)

    for t in range(T - 1, -1, -1):

        t_tensor = torch.full((npts, 1), t, dtype=torch.float32, device=device)

        eps_theta = net(x_t, t_tensor)

        alpha_t = alphas[t]
        alphabar_t = alphabars[t]
        alphabar_prev = alphabars[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t = betas[t]

        # x0 prediction
        x0_hat = (x_t - torch.sqrt(1 - alphabar_t) * eps_theta) / torch.sqrt(alphabar_t)

        # posterior mean
        coef1 = torch.sqrt(alphabar_prev) * beta_t / (1 - alphabar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alphabar_prev) / (1 - alphabar_t)
        mu = coef1 * x0_hat + coef2 * x_t

        var = beta_t * (1 - alphabar_prev) / (1 - alphabar_t)
        sigma = torch.sqrt(var)

        if t > 0:
            x_t = mu + sigma * torch.randn_like(x_t)
        else:
            x_t = mu

        locations[:, :, t] = x_t.cpu().numpy()

print("Trajectory shape:", locations.shape)

# -------------------
# Persistence
# -------------------
def compute_dgms(X):
    return ripser(X, maxdim=1)["dgms"]

# -------------------
# FIXED LAYOUT SETTINGS
# -------------------
X_LIM = (0.0, 2.0)
Y_LIM = (0, 400)  # fixed upper bound

colors = ["C0", "C1"]

def plot_barcodes(ax, dgms, t_label):

    ax.clear()

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    ax.set_xlabel("Filtration value")
    ax.set_title(f"Persistence Barcodes — t = {t_label}")

    y = 0

    for dim, dgm in enumerate(dgms):
        for (birth, death) in dgm:

            if death == np.inf:
                death = X_LIM[1]

            ax.hlines(
                y=y,
                xmin=birth,
                xmax=death,
                color=colors[dim % 2],
                linewidth=1.6
            )
            y += 1

        y += 15  # fixed separation between H0 and H1


# -------------------
# Animation
# -------------------
fig, ax = plt.subplots(figsize=(7, 4))

def update(frame_t):

    # IMPORTANT:
    # frame_t = 0 → data
    # frame_t = T → noise

    t = T - frame_t

    X = locations[:, :, t]
    dgms = compute_dgms(X)

    plot_barcodes(ax, dgms, t_label=t)

ani = animation.FuncAnimation(
    fig,
    update,
    frames=T + 1,
    interval=30,
    blit=False
)

plt.show()

# optional save
ani.save("persistence_barcodes_fixed.gif", fps=30, dpi=150)
