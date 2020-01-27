"""
This file is created to test the correctness and the scalability of the
GNMF implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns
import sys
import time
import argparse

sys.path.append(dirname(dirname(abspath(__file__))))

from src.nmf import Nmf
from src.gnmf import Gnmf

def plot(W, width, height, len):
    """
    Plots all basis images
    - W     : the basis matrix
    - width : # cols in the canvas
    - height: # rows in the canvas
    - len   : the height/width of the matrix (assuming it's a square)
    """
    #TODO: better parameter names
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
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Test GNMF implementation given parameters")
    parser.add_argument("-n", "--height", type=int, help="The height of the V matrix", required=True)
    parser.add_argument("-m", "--width", type=int, help="The width of the V matrix", required=True)
    parser.add_argument("-k", "--rank", type=int, help="The rank used for the GNMF", required=True)
    parser.add_argument("-p", "--pneighbor", type=int, default=5, help="The number of nearest neighbors to be considered")
    parser.add_argument("-l", "--lmbda", type=int, default=10, help="The lambda used for the regularizer")
    parser.add_argument("-i", "--iters", type=int, default=[100], nargs="+", help="The list of # iterations to be run")
    parser.add_argument("-mt", "--method", type=str, default="euclidean", help="The update method: divergence or euclidean")
    input = parser.parse_args()

    # initiate X
    X = np.random.rand(input.height, input.width)
    # initiate gnmf
    gnmf = Gnmf(rank=input.rank, p=input.pneighbor, lmbda=input.lmbda, method=input.method)

    # run given # iterations
    for iter in input.iters:
        time_cp1 = time.time()
        U, V = gnmf.factorize(X, n_iter=iter)
        time_cp2 = time.time()
        print("Total time: ", time_cp2 - time_cp1)
        print("=================================")
