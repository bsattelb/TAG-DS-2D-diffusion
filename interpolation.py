import torch
import numpy as np
import matplotlib.pyplot as plt

import network
from train_network import Scheduler


# -----------------------------------
# Line transport visualization
# -----------------------------------

@torch.no_grad()
def line_transport_figure(
    model,
    scheduler,
    device="cpu",
    sigma_scale=0.0,
    n_points=400,
    line_length=6.0,
    n_panels=24,
    n_rows=3,          # FIXED: always 3 rows
    save_path="line_transport.png"
):

    T = scheduler.T
    AX_LIM = 2.0

    # -----------------------------------
    # initial line in noise space
    # -----------------------------------

    xs = np.linspace(
        -line_length / 2,
        line_length / 2,
        n_points
    )

    line = np.stack(
        [xs, np.zeros_like(xs)],
        axis=1
    )

    trajectory = np.zeros(
        (n_points, 2, T + 1),
        dtype=np.float32
    )

    trajectory[:, :, T] = line

    x = torch.tensor(
        line,
        dtype=torch.float32,
        device=device
    )

    # -----------------------------------
    # reverse diffusion
    # -----------------------------------

    for t in range(T - 1, -1, -1):

        x = model.reverse_diffuse(
            x,
            t,
            sigma_scale=sigma_scale
        )

        trajectory[:, :, t] = (
            x.detach()
             .cpu()
             .numpy()
        )

    # -----------------------------------
    # choose evenly spaced timesteps
    # -----------------------------------

    panel_times = np.linspace(
        T,
        0,
        n_panels
    ).astype(int)

    panel_times = np.unique(panel_times)[::-1]
    n_panels = len(panel_times)

    # force 3 rows
    n_cols = int(np.ceil(n_panels / n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    colors = np.linspace(0, 1, n_points)

    for i, t in enumerate(panel_times):

        ax = axes[i]
        pts = trajectory[:, :, t]

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="black",
            linewidth=1
        )

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors,
            cmap="viridis",
            s=6
        )

        ax.set_title(f"t = {t}")

        # fixed scale
        ax.set_xlim(-AX_LIM, AX_LIM)
        ax.set_ylim(-AX_LIM, AX_LIM)
        ax.set_aspect("equal")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------------
# Main
# -----------------------------------

def main():

    device = "cpu"

    T = 1000
    scheduler = Scheduler(T)

    model = network.Network(
        [3, 100, 100, 100, 2],
        scheduler
    ).to(device)

    checkpoint = torch.load(
        "network_final.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["weights"])
    model.eval()

    line_transport_figure(
        model=model,
        scheduler=scheduler,
        device=device,
        sigma_scale=0.0,
        n_points=400,
        line_length=6.0,
        n_panels=24,
        n_rows=3,   # ← enforced 3 rows
        save_path="line_transport_3rows.png"
    )


if __name__ == "__main__":
    main()
