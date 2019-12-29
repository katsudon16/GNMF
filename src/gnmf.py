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

def read_dataset():
    """
    Read COIL20 dataset, resize to 32x32, and return them as np array X
    where X is an [M x N] matrix; N ~ # data, M ~ dimensions
    """
    coil20_dir = join(dirname(dirname(abspath(__file__))), "dataset", "COIL20")
    coil20_len = 10 # 20
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

def get_weights_matrix(X, p=3):
    """
    Generate weights matrix by p-nearest neighbors + dot-product weighting
    Complexity: O(N^2 * M)
    """
    m, n = X.shape
    # initialize distance matrix and W matrix
    dist_matrix = np.full((n, n), np.inf)
    W = np.zeros((n, n))
    #TODO: implementation improvement?
    # src: https://ljvmiranda921.github.io/notebook/2017/02/09/k-nearest-neighbors/
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist_matrix[i][j] = dist_matrix[j][i] = np.linalg.norm(X[:,i] - X[:,j])
    # finding p-nearest neighbors for each data point
    sorted_idx = np.argsort(dist_matrix, axis=1)
    for i in range(n):
        for j in range(p):
            neighbor = sorted_idx[i][j]
            # compute dot-product weighting
            W[i][neighbor] = np.dot(X[:,i], X[:,neighbor])
    return(W)

def init_rand_matrix(nrow, ncol, seed=None):
    """
    Initialize matrix (random) given # rows and # cols
    """
    if not seed:
        seed = np.random.randint(1000)
    np.random.seed(seed)
    return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

def update_euclidean(X, U, V, W, lmbda=100):
    """
    Update U & V using multiplicative euclidean approach
    """
    # update V
    # calc D
    D = np.diag(np.sum(W, axis=0))
    L = D - W
    V = V * np.divide(U.T @ X + lmbda * (V @ W), U.T @ U @ V + lmbda * (V @ D))
    # update U
    U = U * np.divide(X @ V.T, U @ V @ V.T)
    # calc objective func
    R = X - (U @ V)
    D = np.sum(R * R) + np.trace(lmbda * (V @ L @ V.T))
    return(U, V, D)

def factorize_gnmf(X, rank=10, n_iter = 100, method="euclidean"):
    """
    Factorize matrix X into W and H given rank using multiplicative method
    method options: ["euclidean", "divergence"]
    """
    n, m = X.shape
    print("generating weight matrix...")
    W = get_weights_matrix(X)
    U = init_rand_matrix(n, rank)
    V = init_rand_matrix(rank, m)
    print("starting the iteration...")
    for iter in range(n_iter):
        U, V, obj_val = update_euclidean(X, U, V, W)
        print("Iteration %d; objective value = %.2f" % (iter, obj_val))
    return(U, V)

def run_gnmf():
    """
    Run graph regularized nonnegative matrix factorization

    X ~ data matrix                --> M x N
    W ~ weight matrix              --> N x N
    U ~ basis matrix               --> M x k
    V ~ data representation matrix --> k x N (instead of N x k!)

    Currently uses COIL20 dataset for testing
    """
    X = read_dataset()
    U, V = factorize_gnmf(X)
    plot(U)

def plot(U):
    """
    Plots all 10 basis images
    """
    # sns.heatmap(U)
    # plt.show()
    # return
    plt.set_cmap("gray")
    canvas = Image.new("L", (5 * 32 + 6, 2 * 32 + 3)) # (width, height)
    # 4 rows, 5 cols
    for i in range(2):
        for j in range(5):
            basis = U[:, i * 5 + j].reshape((32, 32))
            basis = basis / basis.max() * 255
            img = expand(Image.fromarray(basis), border=1, fill=255)
            canvas.paste(img.copy(), (j * 32 + j, i * 32 + i))
    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    run_gnmf()
