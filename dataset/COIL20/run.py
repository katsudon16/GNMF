import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns

import sys
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import argparse

from src.nmf import Nmf
from src.gnmf import Gnmf

def read_dataset():
    """
    Read COIL20 dataset, resize to 32x32, and return them as np array X
    """
    coil20_dir = dirname(abspath(__file__))
    coil20_len = 20
    coil20_obj_len = 72
    img_size = 32

    X = np.zeros((img_size * img_size, coil20_len * coil20_obj_len))

    for obj_i in range(coil20_len):
        for obj_img_j in range(coil20_obj_len):
            # open and resize the image
            img = Image.open(join(coil20_dir, "obj%d__%d.png" % (obj_i + 1, obj_img_j))).resize((img_size, img_size))
            img_n = obj_i * coil20_obj_len + obj_img_j
            X[:, img_n] = np.asarray(img).flatten()
    return(X)

def plot(U):
    """
    Plots all 20 basis images
    """
    # sns.heatmap(U)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (5 * 32 + 6, 4 * 32 + 5)) # (w, h)
    # 4 rows, 5 cols
    for i in range(4):
        for j in range(5):
            basis = U[:, i * 5 + j].reshape((32, 32))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * 32 + j, i * 32 + i))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run GNMF on COIL20 dataset")
    parser.add_argument("-k", "--rank", type=int, help="The rank used for the GNMF", required=True)
    parser.add_argument("-p", "--pneighbor", type=int, default=5, help="The number of nearest neighbors to be considered")
    parser.add_argument("-l", "--lmbda", type=int, default=10, help="The lambda used for the regularizer")
    parser.add_argument("-i", "--iters", type=int, default=100, help="The # iterations to be run")
    parser.add_argument("-mt", "--method", type=str, default="euclidean", help="The update method: divergence or euclidean")
    input = parser.parse_args()

    X = read_dataset()
    gnmf = Gnmf(X=X, rank=input.rank, p=input.pneighbor, lmbda=input.lmbda, method=input.method)
    U, V = gnmf.factorize(n_iter=input.iters)
    # nmf = Nmf(rank=input.rank, method=input.method)
    # U, V = nmf.factorize(X, n_iter=input.iters)
    
