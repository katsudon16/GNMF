"""
This file is created to test the correctness and the scalability of the
GNMF implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath, exists
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns
import sys
import time
import argparse

sys.path.append(dirname(dirname(abspath(__file__))))

from src.nmf import Nmf
from src.gnmf import Gnmf

def plot(U, cols, rows, len):
    """
    Plots all basis images
    - U     : the basis matrix
    - cols  : # cols in the canvas
    - rows  : # rows in the canvas
    - len   : the height/width of the matrix (assuming it's a square)
    """
    # sns.heatmap(U)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (cols * len + cols+1, rows * len + rows+1)) # (w, h)
    # 4 rows, 5 cols
    for i in range(rows):
        for j in range(cols):
            basis = U[:, i * cols + j].reshape((len, len))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * len + j, i * len + i))
    plt.imshow(canvas)
    plt.show()

def calc_reconstruction_error(X, U, V):
    X_new = U @ V
    return(np.sum((X_new - X)**2))

if __name__ == "__main__":
    # retrieve X & W from test dataset numpy files
    np_dir = join(dirname(abspath(__file__)), "dataset")
    heights = [100, 500, 1000, 5000, 10000, 30000]
    widths = [50, 100, 500, 1000, 3000, 5000]
    pneighbors = [3, 5, 7, 10, 15, 20, 50]
    n_iter = 500

    generate_W = True

    # location for saving results
    res_dir = join(dirname(abspath(__file__)), "results")
    results = []

    for h in heights:
        for w in widths:
            for p in pneighbors:
                if not exists(join(np_dir, "h%d_w%d_p%d_X.npy" % (h, w, p))):
                    continue
                X = np.load(join(np_dir, "h%d_w%d_p%d_X.npy" % (h, w, p)))

                W = (None if generate_W else
                        np.load(join(np_dir, "h%d_w%d_p%d_W.npy" % (h, w, p))))

                #TODO: test with different ranks and lambdas
                rank = 10
                lmbda = 10

                # run gnmf
                gnmf = Gnmf(X=X, rank=rank, W=W, lmbda=lmbda, method="euclidean")
                time_cp1 = time.time()
                gnmf_U, gnmf_V, obj_vals = gnmf.factorize(n_iter=n_iter, return_obj_values=True)
                time_cp2 = time.time()
                time_gnmf = time_cp2 - time_cp1

                # output plot obj_vals >< iterations
                sns_plot = sns.lineplot(x=range(1, 501), y=obj_vals)
                sns_plot.set(title="%dx%d matrix; k=%d; p=%d; lambda=%d" % (h, w, rank, p, lmbda),
                             xlabel="Iterations", ylabel="Objective function values")
                plt.savefig(join(res_dir, "gnmf_%dx%d_k%d_p%d_ld%d.png" % (h, w, rank, p, lmbda)))
                plt.clf()

                # compute reconstruction errors
                gnmf_error = calc_reconstruction_error(X, gnmf_U, gnmf_V)

                # run nmf
                nmf = Nmf(rank=rank, method="euclidean")
                nmf_U, nmf_V = nmf.factorize(X, n_iter=n_iter)
                nmf_error = calc_reconstruction_error(X, nmf_U, nmf_V)
                results.append([h, w, rank, p, lmbda, time_gnmf, gnmf_error, nmf_error])

    with open(join(res_dir, "results.csv"), "w") as file:
        file.write("n,m,k,p,lambda,gnmf_dur,gnmf_error,nmf_error\n")
        for res in results:
            file.write("%s\n" % ",".join(str(x) for x in res))
