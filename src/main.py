import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image

# 1. compute weights
# 2. compute smoothness of the low-dimensional representation
# 3. compute objective functions
# 4. build update functions

# def factorize_nmf(V):

def init_matrix():
    """
    Initialize matrix (random)
    """

def read_dataset():
    """
    Read COIL20 dataset, resize to 32x32, and return them as np array V
    """
    print(dirname(dirname(os.getcwd())))
    coil20_dir = join(dirname(dirname(abspath(__file__))), "dataset", "COIL20")
    coil20_len = 20
    coil20_obj_len = 72
    img_size = 32
    V = np.zeros((img_size * img_size, coil20_len * coil20_obj_len))

    for obj_i in range(coil20_len):
        for obj_img_j in range(coil20_obj_len):
            # open and resize the image
            img = Image.open(join(coil20_dir, "obj%d__%d.png" % (obj_i + 1, obj_img_j))).resize((img_size, img_size))
            img_n = obj_i * coil20_obj_len + obj_img_j
            V[:, img_n] = np.asarray(img).flatten()

    return(V)

def factorize_nmf(V, rank=20):
    """
    Factorizes matrix V into W and H given rank
    """
    W = init_W()
    H = init_H()

def run_nmf():
    """
    Run graph regularized nonnegative matrix factorization

    Currently uses COIL20 dataset for testing
    """
    V = read_dataset()
    #  = preprocess(V) # normalize
    W, H = factorize_nmf(V)
    # plot(W)

# def plot():
    # set_cmap('gray')
    # blank = new("L", (133 + 6, 133 + 6))
    # for i in range(7):
    #     for j in range(7):
    #         basis = np.array(W[:, 7 * i + j])[:, 0].reshape((19, 19))
    #         basis = basis / np.max(basis) * 255
    #         basis = 255 - basis
    #         ima = fromarray(basis)
    #         ima = ima.rotate(180)
    #         expand(ima, border=1, fill='black')
    #         blank.paste(ima.copy(), (j * 19 + j, i * 19 + i))
    # imshow(blank)
    # savefig("cbcl_faces.png")

if __name__ == "__main__":
    run_nmf()
