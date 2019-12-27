import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, dirname, abspath
from PIL import Image
from PIL.ImageOps import expand
import seaborn as sns

# 1. compute weights
# 2. compute smoothness of the low-dimensional representation
# 3. compute objective functions
# 4. build update functions

# def factorize_nmf(V):

def init_rand_matrix(nrow, ncol, seed=None):
    """
    Initialize matrix (random) given # rows and # cols
    """
    if not seed:
        seed = np.random.randint(1000)
    np.random.seed(seed)
    return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

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

def factorize_nmf(V, rank=20, n_iter = 100):
    """
    Factorizes matrix V into W and H given rank
    """
    n, m = V.shape
    W = init_rand_matrix(n, rank)
    H = init_rand_matrix(rank, m)
    #TODO: multiplicative or additive (grad descent)?
    #TODO: euclidean or divergence
    # curr: euclidean and multiplicative
    for iter in range(n_iter):
        # update H
        H = H * np.divide(W.T @ V, W.T @ W @ H)
        # update W
        W = W * np.divide(V @ H.T, W @ H @ H.T)
        # calc objective func
        R = V - (W @ H)
        D = np.sum(R * R)
        print("Current objective function value", D)

    return(W, H)

def run_nmf():
    """
    Run graph regularized nonnegative matrix factorization

    Currently uses COIL20 dataset for testing
    """
    V = read_dataset()
    #  = preprocess(V) # normalize
    W, H = factorize_nmf(V)
    plot(W)

def plot(W):
    """
    Plots all 20 basis images
    """
    plt.set_cmap("gray")
    canvas = Image.new("L", (5 * 32 + 6, 4 * 32 + 5)) # (w, h)
    # 4 rows, 5 cols
    for i in range(4):
        for j in range(5):
            basis = W[:, i * 5 + j].reshape((32, 32))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * 32 + j, i * 32 + i))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    run_nmf()
