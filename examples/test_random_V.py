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

def plot(W, width, height):
    """
    Plots all 20 basis images
    """
    # sns.heatmap(W)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (width * 5 + width+1, height * 5 + height+1)) # (w, h)
    # 4 rows, 5 cols
    for i in range(height):
        for j in range(width):
            basis = W[:, i * width + j].reshape((5, 5))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * 5 + j, i * 5 + i))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    n = 25
    m = 100
    V = np.random.rand(n, m)
    rank = 5
    nmf = Nmf(method="divergence")
    W, H = nmf.factorize(V, rank=rank, n_iter=100)
    plot(W, 5, 1)
