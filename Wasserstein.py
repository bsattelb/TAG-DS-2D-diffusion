import torch
import numpy as np
import network
import train_network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ot


def run_sample(net, device, mode, n=500):
    """
    mode:
        0 = DDIM
        1 = DDPM
    """
    pts = net.sample(n, mode, device)

    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()

    return pts


def main():
    device = "cpu"

    network_shape = [3, 100, 100, 100, 2]
    scheduler = train_network.Scheduler()

    net = network.Network(network_shape, scheduler).to(device)

    checkpoint = torch.load("network_final.pt", map_location=device)
    net.load_state_dict(checkpoint["weights"])
    net.eval()

    n = 500

    # =========================================================
    # SAMPLE BOTH MODELS
    # =========================================================

    print("Sampling DDIM...")
    ddim_pts = run_sample(net, device, mode=0, n=n)

    print("Sampling DDPM...")
    ddpm_pts = run_sample(net, device, mode=1, n=n)

    print("DDIM shape:", ddim_pts.shape)
    print("DDPM shape:", ddpm_pts.shape)

    np.save("ddim_points.npy", ddim_pts)
    np.save("ddpm_points.npy", ddpm_pts)

    print("Saved ddim_points.npy and ddpm_points.npy")

    T = ddim_pts.shape[2]

    # =========================================================
    # ANIMATION
    # =========================================================

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_ddim, ax_ddpm = axes

    for ax, title in zip(axes, ["DDIM", "DDPM"]):
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect("equal")
        ax.set_title(title)

    scat_ddim = ax_ddim.scatter(ddim_pts[:, 0, T - 1], ddim_pts[:, 1, T - 1], s=2)
    scat_ddpm = ax_ddpm.scatter(ddpm_pts[:, 0, T - 1], ddpm_pts[:, 1, T - 1], s=2)

    def update(i):
        t = T - 1 - i
        scat_ddim.set_offsets(np.c_[ddim_pts[:, 0, t], ddim_pts[:, 1, t]])
        scat_ddpm.set_offsets(np.c_[ddpm_pts[:, 0, t], ddpm_pts[:, 1, t]])
        ax_ddim.set_title(f"DDIM t={t}")
        ax_ddpm.set_title(f"DDPM t={t}")
        return scat_ddim, scat_ddpm

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=True)
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 2-WASSERSTEIN OVER CHUNKS (GAUSSIAN → SAMPLE)
    # =========================================================

    import ot

    DELTA = 1  # compare t with t-DELTA

    def wasserstein_chunked_forward(pts, delta):
        T = pts.shape[2]
        n = pts.shape[0]

        a = np.ones(n) / n
        b = np.ones(n) / n

        times = []
        w = []

        for t in range(T - 1, delta - 1, -delta):
            x = pts[:, :, t]  # current step
            y = pts[:, :, t - delta]  # next delta step toward data

            M = ot.dist(x, y)
            w.append(np.sqrt(ot.emd2(a, b, M)))

            times.append(t)  # plot at the starting timestep

        return np.array(times), np.array(w)

    print(f"Computing chunked Wasserstein distances, Δ={DELTA}...")

    times_ddim, w_ddim = wasserstein_chunked_forward(ddim_pts, DELTA)
    times_ddpm, w_ddpm = wasserstein_chunked_forward(ddpm_pts, DELTA)

    # shared y-limits
    ymin = min(w_ddim.min(), w_ddpm.min())
    ymax = max(w_ddim.max(), w_ddpm.max())

    # =========================================================
    # DDIM
    # =========================================================

    plt.figure(figsize=(7, 4))  # wider figure

    plt.plot(times_ddim, w_ddim, label="DDIM")
    plt.scatter(times_ddim, w_ddim, s=20)

    plt.title(f"DDIM: W2 per step (Δ={DELTA})")
    plt.xlabel("timestep (1000 → 0)")
    plt.ylabel("2-Wasserstein distance")

    # Fit y-axis tightly to the data
    plt.ylim(w_ddim.min() - 0.02 * abs(w_ddim.min()),
             w_ddim.max() + 0.02 * abs(w_ddim.max()))

    # x-axis run 1000 → 0 left→right
    plt.xlim([times_ddim.max() + 10, times_ddim.min() - 10])  # wider x-axis

    plt.tight_layout()
    plt.show()

    # =========================================================
    # DDPM
    # =========================================================

    plt.figure(figsize=(7, 4))  # wider figure

    plt.plot(times_ddpm, w_ddpm, label="DDPM")
    plt.scatter(times_ddpm, w_ddpm, s=20)

    plt.title(f"DDPM: W2 per step (Δ={DELTA})")
    plt.xlabel("timestep (1000 → 0)")
    plt.ylabel("2-Wasserstein distance")

    # Fit y-axis tightly to the data
    plt.ylim(w_ddpm.min() - 0.02 * abs(w_ddpm.min()),
             w_ddpm.max() + 0.02 * abs(w_ddpm.max()))

    # x-axis run 1000 → 0 left→right
    plt.xlim([times_ddpm.max() + 10, times_ddpm.min() - 10])  # wider x-axis

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
