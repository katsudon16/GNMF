import numpy as np

class Gnmf(object):
    """
    Graph regularized NMF
    """
    def __init__(self, rank=10, p=3, method="euclidean"):
        """
        - rank  : NMF rank
        - p     : # closest neighbors to be taken into account in the weight matrix
        - method: "euclidean" or "divergence"
        """
        self.rank = rank
        self.p = p
        self.method = method

    def init_rand_matrix(self, nrow, ncol, seed=None):
        """
        Initialize matrix (random) given # rows and # cols
        """
        if not seed:
            seed = np.random.randint(1000)
        np.random.seed(seed)
        return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

    def get_weights_matrix(self, X):
        """
        Generate weights matrix by p-nearest neighbors + dot-product weighting
        Complexity: O(N^2 * M)
        """
        m, n = X.shape
        p = self.p
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
                #TODO: ensure W is symmetric?
                W[i][neighbor] = W[neighbor][i] = np.dot(X[:,i], X[:,neighbor])
        return(W)

    def update_euclidean(self, X, U, V, W, lmbda=100):
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
        D = np.sum(R * R) + lmbda * np.trace(V @ L @ V.T)
        return(U, V, D)

    def factorize(self, X, n_iter=100, method="euclidean"):
        """
        Factorize matrix X into W and H given rank using multiplicative method
        method options: ["euclidean", "divergence"]
        """
        n, m = X.shape
        rank = self.rank
        print("generating weight matrix...")
        W = self.get_weights_matrix(X)
        U = self.init_rand_matrix(n, rank)
        V = self.init_rand_matrix(rank, m)
        print("starting the iteration...")
        for iter in range(n_iter):
            U, V, obj_val = self.update_euclidean(X, U, V, W)
            print("Iteration %d; objective value = %.2f" % (iter, obj_val))
        # set the euclidean length of each col vec in U = 1
        sum_col_U = np.sqrt(np.sum(U**2, axis=0))
        U = U / sum_col_U
        V = V / sum_col_U.reshape((rank, 1))
        return(U, V)
