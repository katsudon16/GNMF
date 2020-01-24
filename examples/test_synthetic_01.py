import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns

import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from src.nmf import Nmf
from src.gnmf import Gnmf

def plot(W, width, height, len):
    """
    Plots all 20 basis images
    """
    # sns.heatmap(W)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (width * len + width+1, height * len + height+1)) # (w, h)
    # 4 rows, 5 cols
    for i in range(height):
        for j in range(width):
            basis = W[:, i * width + j].reshape((len, len))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * len + j, i * len + i))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    n = 9
    m = 25
    rank = 4

    W = np.array([
    [250, 0, 0, 250],
    [250, 0, 0, 0],
    [250, 250, 0, 0],
    [0, 250, 0, 0],
    [0, 250, 250, 0],
    [0, 0, 250, 0],
    [0, 0, 250, 250],
    [0, 0, 0, 250],
    [0, 0, 0, 250]
    ])
    H = np.random.rand(rank, m)
    V = W @ H
    # nmf = Nmf(rank=rank, method="euclidean")
    # W, H = nmf.factorize(V, n_iter=100)
    # plot(W, 5, 1)
    gnmf = Gnmf(p=2, lmbda=20, method="euclidean")
    W, H = gnmf.factorize(V, n_iter=100)
    plot(W, 4, 1, 3)
