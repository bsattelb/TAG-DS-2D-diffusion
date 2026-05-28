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

# -------------------
# Generate sample trajectory
# -------------------
with torch.no_grad():
    locations = net.sample(
        npts=800,
        sigma_scale=1,
        device=device
    )

# =========================================================
# IMPORTANT: FIX TIME CONVENTION ONCE AND FOR ALL
# Now we enforce:
#   index 0   = Gaussian noise
#   index T   = learned data distribution
# =========================================================

locations = locations[:, :, ::-1]

# -------------------
# OPTIONAL: convert to list format (only for printing/debugging)
# -------------------
trajectory_list = []

for t in range(locations.shape[2]):
    coords_t = locations[:, :, t]

    coords_list = [
        (float(x), float(y))
        for x, y in coords_t
    ]

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

def update(t):
    ax.clear()

    coords = locations[:, :, t]

    ax.scatter(coords[:, 0], coords[:, 1], s=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    if t == 0:
        title = "Gaussian noise"
    elif t == locations.shape[2] - 1:
        title = "Learned data distribution"
    else:
        title = f"t = {t}"

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
# DIAGNOSTIC PLOTS (CORRECTED)
# -------------------

# Gaussian (start)
plt.figure()
plt.scatter(locations[:,0,0], locations[:,1,0], s=5)
plt.title("t = 0 (Gaussian noise)")
plt.axis("equal")
plt.show()

# Data (end)
plt.figure()
plt.scatter(locations[:,0,-1], locations[:,1,-1], s=5)
plt.title("t = T (learned distribution)")
plt.axis("equal")
plt.show()