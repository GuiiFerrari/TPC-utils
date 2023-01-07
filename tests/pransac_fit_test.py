import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import pRansac, fit_3D
import os


def load_data(path):
    input = np.load(path + "/evento_teste.npy")
    return input


if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "..", "data")
    pc = load_data(path)
    inliers, versors, points = pRansac(pc, 800, 24, 15, mode=3)
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.set_zlim((-140, 140))
    ax1.set_xlim((-140, 140))
    ax1.set_ylim((-0, 512))
    ax1.set_title("pRansac")
    colors = plt.cm.Spectral(np.linspace(0, 1, len(points)))
    # Test pRansac
    for index, i, v, p in zip(np.arange(len(inliers)), inliers, versors, points):
        linepts = v * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += p
        x0, y0, z0, x1, y1, z1 = (
            linepts[0][0],
            linepts[0][1],
            linepts[0][2],
            linepts[1][0],
            linepts[1][1],
            linepts[1][2],
        )
        ax1.scatter3D(pc[i][:, 0], pc[i][:, 2], pc[i][:, 1], color=colors[index], s=1)
        ax1.plot3D(
            np.array([x0, x1]),
            np.array([z0, z1]),
            np.array([y0, y1]),
            alpha=0.75,
            label=f"Line number {index}",
            color=colors[index],
        )
    ax1.legend()
    plt.show()

    # Test fit_3D
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.set_title("3D fit")
    for index, i in enumerate(inliers):
        v, p = fit_3D(pc[i])
        linepts = v * np.mgrid[-800:800:2j][:, np.newaxis]
        linepts += p
        x0, y0, z0, x1, y1, z1 = (
            linepts[0][0],
            linepts[0][1],
            linepts[0][2],
            linepts[1][0],
            linepts[1][1],
            linepts[1][2],
        )
        ax1.scatter3D(pc[i][:, 0], pc[i][:, 2], pc[i][:, 1], color=colors[index], s=1)
        ax1.plot3D(
            np.array([x0, x1]),
            np.array([z0, z1]),
            np.array([y0, y1]),
            alpha=0.75,
            label=f"Line number {index}",
            color=colors[index],
        )
    ax1.legend()
    plt.show()
