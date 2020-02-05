"""
This file is created to test the correctness and the scalability of the
GNMF implementation.
"""

import numpy as np
import os
from os.path import join, dirname, abspath
import sys
import argparse
import time

sys.path.append(dirname(dirname(abspath(__file__))))

from src.gnmf import Gnmf

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Generate X and W matrices for GNMF testing")
    parser.add_argument("-n", "--height", type=int, help="The height of the V matrix", required=True)
    parser.add_argument("-m", "--width", type=int, help="The width of the V matrix", required=True)
    parser.add_argument("-p", "--pneighbor", type=int, default=5, help="The number of nearest neighbors to be considered")
    parser.add_argument("-s", "--seed", type=int, default=None, help="The random seed")
    input = parser.parse_args()
    print(input)

    output_file_X = join(dirname(abspath(__file__)), "dataset", "h%d_w%d_p%d_X.npy" % (input.height, input.width, input.pneighbor))
    output_file_W = join(dirname(abspath(__file__)), "dataset", "h%d_w%d_p%d_W.npy" % (input.height, input.width, input.pneighbor))
    output_file_info = join(dirname(abspath(__file__)), "dur_generate_matrix.csv")

    # initiate X
    if input.seed:
        np.random.seed(input.seed)
    X = np.random.rand(input.height, input.width)

    # initiate gnmf to retrieve W matrix
    time_cp1 = time.time()
    gnmf = Gnmf(X=X, p=input.pneighbor)
    time_cp2 = time.time()

    W = gnmf.get_weight_matrix()

    # store X and W
    np.save(output_file_X, X)
    np.save(output_file_W, W)

    # store the total time for generating the weight matrix
    with open(output_file_info, "a") as file:
        file.write("%d,%d,%d,%d\n" % (input.height, input.width, input.pneighbor, time_cp2-time_cp1))
