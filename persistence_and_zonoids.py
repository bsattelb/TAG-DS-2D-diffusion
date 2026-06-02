import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import network
import torch

from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from ripser import ripser
from persim import plot_diagrams
from shapely.geometry import Polygon
from train_network import Scheduler

###########################################################################
# DIFFUSION TRAJECTORY EXTRACTION
###########################################################################

def generate_diffusion_trajectory(
        model,
        npts,
        sigma_scale,
        device):

    T = model.scheduler.T

    locations = np.zeros(
        (npts, 2, T + 1),
        dtype=np.float32
    )

    vec_fields = np.zeros(
        (npts, 2, T),
        dtype=np.float32
    )

    x0 = np.random.randn(npts, 2)

    locations[:, :, T] = x0

    with torch.no_grad():

        xs = torch.tensor(
            x0,
            dtype=torch.float32,
            device=device
        )

        for t in range(T - 1, -1, -1):

            xs, vf = model.reverse_diffuse(
                xs,
                t,
                sigma_scale=sigma_scale,
                return_vec_field=True
            )

            locations[:, :, t] = (
                xs.detach()
                  .cpu()
                  .numpy()
            )

            vec_fields[:, :, t] = (
                vf.detach()
                  .cpu()
                  .numpy()
            )

    return locations, vec_fields


###########################################################################
# PERSISTENT HOMOLOGY
###########################################################################

def compute_persistence_sequence(
        locations,
        maxdim=1):

    T = locations.shape[2]

    diagrams = []

    for t in tqdm(
            range(T),
            desc="Persistence"):

        cloud = locations[:, :, t]

        dgms = ripser(
            cloud,
            maxdim=maxdim
        )["dgms"]

        diagrams.append(dgms)

    return diagrams


def save_barcode_frames(
        diagrams,
        outdir):

    os.makedirs(outdir, exist_ok=True)

    for t, dgm in enumerate(diagrams):

        fig = plt.figure(figsize=(7, 5))

        plot_diagrams(
            dgm,
            show=False
        )

        plt.title(f"Persistence t={t}")

        plt.savefig(
            os.path.join(
                outdir,
                f"barcode_{t:04d}.png"
            )
        )

        plt.close()


###########################################################################
# VITALE ZONOID
###########################################################################

def approximate_vitale_zonoid(
        cloud,
        k_vectors=100,
        n_samples=10000):

    idx = np.random.choice(
        len(cloud),
        size=k_vectors,
        replace=True
    )

    V = cloud[idx]

    lambdas = np.random.rand(
        n_samples,
        k_vectors
    )

    Z = lambdas @ V

    Z /= k_vectors

    return Z


def convex_hull_polygon(points):

    hull = ConvexHull(points)

    verts = points[hull.vertices]

    return verts


def polygon_area(vertices):

    poly = Polygon(vertices)

    return poly.area


###########################################################################
# HAUSDORFF DISTANCE
###########################################################################

def hausdorff_distance(A, B):

    D = cdist(A, B)

    h1 = np.max(np.min(D, axis=1))
    h2 = np.max(np.min(D, axis=0))

    return max(h1, h2)


###########################################################################
# EVOLVING VITALE ZONOIDS
###########################################################################

def compute_zonoid_sequence(
        locations,
        k_vectors=100,
        n_samples=10000):

    T = locations.shape[2]

    zonoids = []
    hulls = []
    areas = []

    for t in tqdm(
            range(T),
            desc="Point-cloud zonoids"):

        cloud = locations[:, :, t]

        Z = approximate_vitale_zonoid(
            cloud,
            k_vectors,
            n_samples
        )

        hull = convex_hull_polygon(Z)

        zonoids.append(Z)
        hulls.append(hull)
        areas.append(
            polygon_area(hull)
        )

    return zonoids, hulls, np.array(areas)


###########################################################################
# VECTOR FIELD ZONOIDS
###########################################################################

def compute_vectorfield_zonoids(
        vec_fields,
        k_vectors=100,
        n_samples=10000):

    T = vec_fields.shape[2]

    zonoids = []
    hulls = []
    areas = []

    for t in tqdm(
            range(T),
            desc="Vector-field zonoids"):

        vectors = vec_fields[:, :, t]

        Z = approximate_vitale_zonoid(
            vectors,
            k_vectors,
            n_samples
        )

        hull = convex_hull_polygon(Z)

        zonoids.append(Z)
        hulls.append(hull)
        areas.append(
            polygon_area(hull)
        )

    return zonoids, hulls, np.array(areas)


###########################################################################
# PLOT ZONOID SEQUENCE
###########################################################################

def save_zonoid_frames(
        locations,
        hulls,
        outdir):

    os.makedirs(outdir, exist_ok=True)

    T = len(hulls)

    for t in range(T):

        cloud = locations[:, :, t]

        hull = hulls[t]

        plt.figure(figsize=(6, 6))

        plt.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=2,
            alpha=0.25
        )

        closed = np.vstack(
            [hull, hull[0]]
        )

        plt.plot(
            closed[:, 0],
            closed[:, 1],
            'r',
            lw=3
        )

        plt.axis("equal")

        plt.title(
            f"Vitale Zonoid t={t}"
        )

        plt.savefig(
            os.path.join(
                outdir,
                f"zonoid_{t:04d}.png"
            )
        )

        plt.close()


###########################################################################
# MOVIES
###########################################################################

def make_movie(
        frame_dir,
        outfile,
        prefix):

    frames = []

    files = sorted([
        f for f in os.listdir(frame_dir)
        if f.startswith(prefix)
    ])

    for f in files:

        frames.append(
            imageio.imread(
                os.path.join(
                    frame_dir,
                    f
                )
            )
        )

    imageio.mimsave(
        outfile,
        frames,
        fps=10
    )


###########################################################################
# HAUSDORFF EVOLUTION
###########################################################################

def successive_hausdorff(
        hulls):

    d = []

    for i in range(len(hulls)-1):

        d.append(
            hausdorff_distance(
                hulls[i],
                hulls[i+1]
            )
        )

    return np.array(d)


###########################################################################
# MASTER PIPELINE
###########################################################################

def diffusion_topology_geometry_pipeline(
        model,
        device,
        npts=5000,
        sigma_scale=0.0,
        outdir="results"):

    os.makedirs(outdir, exist_ok=True)

    print("Generating trajectory")

    locations, vec_fields = \
        generate_diffusion_trajectory(
            model,
            npts,
            sigma_scale,
            device
        )

    #######################################################################
    # Persistence
    #######################################################################

    diagrams = compute_persistence_sequence(
        locations
    )

    save_barcode_frames(
        diagrams,
        os.path.join(
            outdir,
            "barcodes"
        )
    )

    #######################################################################
    # Point cloud Vitale zonoids
    #######################################################################

    zonoids, hulls, areas = \
        compute_zonoid_sequence(
            locations
        )

    save_zonoid_frames(
        locations,
        hulls,
        os.path.join(
            outdir,
            "zonoids"
        )
    )

    #######################################################################
    # Vector field zonoids
    #######################################################################

    vf_zonoids, vf_hulls, vf_areas = \
        compute_vectorfield_zonoids(
            vec_fields
        )

    #######################################################################
    # Hausdorff evolution
    #######################################################################

    hd = successive_hausdorff(hulls)

    plt.figure(figsize=(8,4))
    plt.plot(hd)
    plt.title(
        "Hausdorff Distance Between Successive Zonoids"
    )
    plt.savefig(
        os.path.join(
            outdir,
            "hausdorff.png"
        )
    )
    plt.close()

    #######################################################################
    # Area evolution
    #######################################################################

    plt.figure(figsize=(8,4))
    plt.plot(areas)
    plt.title(
        "Point-cloud Vitale Zonoid Area"
    )
    plt.savefig(
        os.path.join(
            outdir,
            "zonoid_area.png"
        )
    )
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(vf_areas)
    plt.title(
        "Vector-field Vitale Zonoid Area"
    )
    plt.savefig(
        os.path.join(
            outdir,
            "vectorfield_zonoid_area.png"
        )
    )
    plt.close()

    #######################################################################
    # Movies
    #######################################################################

    make_movie(
        os.path.join(
            outdir,
            "barcodes"
        ),
        os.path.join(
            outdir,
            "barcode_evolution.gif"
        ),
        "barcode"
    )

    make_movie(
        os.path.join(
            outdir,
            "zonoids"
        ),
        os.path.join(
            outdir,
            "zonoid_evolution.gif"
        ),
        "zonoid"
    )

    return {
        "locations": locations,
        "vec_fields": vec_fields,
        "diagrams": diagrams,
        "zonoid_hulls": hulls,
        "zonoid_areas": areas,
        "vf_hulls": vf_hulls,
        "vf_areas": vf_areas,
        "hausdorff": hd
    }

if __name__ == "__main__":
    device = "cuda"
    T = 1000
    scheduler = Scheduler(T)

    model = network.Network([3, 100, 100, 100, 2], scheduler).to(device)

    checkpoint = torch.load(
        "network_final.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    results = diffusion_topology_geometry_pipeline(
        model=model,
        device=device,
        npts=500,
        sigma_scale=0.0,   # DDIM
        outdir="results"
    )
