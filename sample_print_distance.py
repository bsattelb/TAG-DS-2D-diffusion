import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import network
from train_network import Scheduler

# -------------------
# Setup
# -------------------
device = 'cpu'

T = 1000
scheduler = Scheduler(T)

net = network.Network([3, 100, 100, 100, 2], scheduler).to(device)

checkpoint = torch.load(
    "checkpoints/nonlinears_e3.pt",
    map_location=device
)
net.load_state_dict(checkpoint["weights"])
net.eval()

# ---- convert scheduler arrays to torch tensors ----
betas = torch.tensor(scheduler.betas, dtype=torch.float32, device=device)
alphas = torch.tensor(scheduler.alphas, dtype=torch.float32, device=device)
alphabars = torch.tensor(scheduler.alphabars, dtype=torch.float32, device=device)

# -------------------
# Generate sample trajectory (DDPM)
# -------------------
npts = 800
locations = np.zeros((npts, 2, T + 1))

# start from pure Gaussian at t = T
x_T = np.random.normal(0, 1, (npts, 2))
locations[:, :, T] = x_T

with torch.no_grad():
    x_t = torch.tensor(x_T, dtype=torch.float32, device=device)

    for t in range(T - 1, -1, -1):
        t_int = t
        # shape (npts, 1) to match your network.forward
        t_tensor = torch.full((npts, 1), t_int, dtype=torch.float32, device=device)

        # 1) predict epsilon
        eps_theta = net(x_t, t_tensor)  # (npts, 2)

        alpha_t = alphas[t_int]
        alphabar_t = alphabars[t_int]
        if t_int > 0:
            alphabar_prev = alphabars[t_int - 1]
        else:
            alphabar_prev = torch.tensor(1.0, device=device)

        beta_t = betas[t_int]

        # 2) estimate x0
        x0_hat = (x_t - torch.sqrt(1 - alphabar_t) * eps_theta) / torch.sqrt(alphabar_t)

        # 3) posterior mean μ_t(x_t, x0_hat)
        coef1 = torch.sqrt(alphabar_prev) * beta_t / (1 - alphabar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alphabar_prev) / (1 - alphabar_t)
        mu = coef1 * x0_hat + coef2 * x_t

        # 4) posterior variance
        var = beta_t * (1 - alphabar_prev) / (1 - alphabar_t)
        sigma = torch.sqrt(var)

        if t_int > 0:
            z = torch.randn_like(x_t)
            x_t = mu + sigma * z
        else:
            x_t = mu  # last step: no noise

        locations[:, :, t_int] = x_t.cpu().numpy()

# -------------------
# OPTIONAL: convert to list format (only for printing/debugging)
# -------------------
trajectory_list = []
for t in range(locations.shape[2]):
    coords_t = locations[:, :, t]
    coords_list = [(float(x), float(y)) for x, y in coords_t]
    trajectory_list.append(coords_list)

# -------------------
# PRINT (optional)
# -------------------
for t, coords in enumerate(trajectory_list):
    print(f"\n===== TIME STEP {t} =====")
    print(coords)

# -------------------
# ANIMATION (Gaussian → Data)
# -------------------
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

def update(frame_t):
    ax.clear()
    coords = locations[:, :, frame_t]
    ax.scatter(coords[:, 0], coords[:, 1], s=5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    if frame_t == T:
        title = "t = T (Gaussian noise)"
    elif frame_t == 0:
        title = "t = 0 (learned distribution)"
    else:
        title = f"t = {frame_t}"
    ax.set_title(title)

ani = animation.FuncAnimation(
    fig,
    update,
    frames=locations.shape[2],
    interval=30,
    blit=False
)

plt.show()

# -------------------
# DIAGNOSTIC PLOTS
# -------------------

# Gaussian (start)
plt.figure()
plt.scatter(locations[:, 0, T], locations[:, 1, T], s=5)
plt.title("t = T (Gaussian noise)")
plt.axis("equal")
plt.show()

# Data (end)
plt.figure()
plt.scatter(locations[:, 0, 0], locations[:, 1, 0], s=5)
plt.title("t = 0 (learned distribution)")
plt.axis("equal")
plt.show()