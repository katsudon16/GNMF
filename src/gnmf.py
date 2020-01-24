import numpy as np

class Gnmf(object):
    """
    Graph regularized NMF
    """
    def __init__(self, rank=10, p=3, lmbda=100, method="euclidean"):
        """
        - rank  : NMF rank
        - p     : # closest neighbors to be taken into account in the weight matrix
        - method: "euclidean" or "divergence"
        """
        self.rank = rank
        self.p = p
        self.method = method
        self.lmbda = lmbda

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
        _, m = X.shape
        p = self.p
        # initialize distance matrix and W matrix
        dist_matrix = np.full((m, m), np.inf)
        W = np.zeros((m, m))
        #TODO: implementation improvement? remove for loops
        # src: https://ljvmiranda921.github.io/notebook/2017/02/09/k-nearest-neighbors/
        #TODO: allow users to construct their own W matrix - test if symmetric
        for i in range(m - 1):
            for j in range(i + 1, m):
                dist_matrix[i][j] = dist_matrix[j][i] = np.linalg.norm(X[:,i] - X[:,j])
        # finding p-nearest neighbors for each data point
        sorted_idx = np.argsort(dist_matrix, axis=1)
        for i in range(m):
            for j in range(p):
                neighbor = sorted_idx[i][j]
                # compute dot-product weighting
                W[i][neighbor] = W[neighbor][i] = np.dot(X[:,i], X[:,neighbor])
        #TODO*: locality sensitive hashing (LSH) - spend few minutes read on finding knn
        return(W)

    def update_euclidean(self, X, U, V, W):
        """
        Update U & V using multiplicative euclidean approach
        """
        lmbda = self.lmbda
        # update V
        # calc the Laplacian matrix L
        D = np.diag(np.sum(W, axis=0))
        L = D - W
        V = V * np.divide(U.T @ X + lmbda * (V @ W), U.T @ U @ V + lmbda * (V @ D))
        # update U
        U = U * np.divide(X @ V.T, U @ V @ V.T)
        # calc objective func
        R = X - (U @ V)
        D = np.sum(R * R) + lmbda * np.trace(V @ L @ V.T)
        return(U, V, D)

    def np_pos(self, np_ar, add_eps=False):
        """Ensures all values in a numpy array > 0"""
        eps = np.finfo(np_ar.dtype).eps
        if add_eps:
            return(np_ar + eps)
        np_ar[np_ar == 0] = eps
        return(np_ar)

    def update_divergence(self, X, U, V, W, lmbda=100):
        """
        Update U & V using multiplicative divergence approach
        """
        n, m = X.shape
        k, _ = V.shape
        # calc the Laplacian matrix L
        D = np.diag(np.sum(W, axis=0))
        L = D - W
        print(np.sum(U), np.sum(V))
        # update V
        #TODO: improve using iterative algorithm CG
        V = V * (U.T @ np.divide(X, U @ V))
        U_row_sum = np.sum(U, axis=0).reshape((k, 1))
        # TODO: check if it's scalable - test with different matrix size - use time package
        for i in range(k):
            # TODO: pseudoinverse pinv
            V[i] = V[i] @ np.linalg.inv(U_row_sum[i] * np.identity(m) + lmbda * L)
        # update U
        V_col_sum = np.sum(V, axis=1).reshape((1, k))
        U = U * np.divide(np.divide(X, U @ V) @ V.T, V_col_sum)
        # calc obj_val
        X_temp = U @ V
        obj_val = np.sum(X * np.log(
        self.np_pos(np.divide(self.np_pos(X), self.np_pos(X_temp)), add_eps=True)
        ) - X + X_temp)
        return(U, V, obj_val)

    def factorize(self, X, n_iter=100):
        """
        Factorize matrix X into W and H given rank using multiplicative method
        method options: ["euclidean", "divergence"]
        """
        n, m = X.shape
        rank = self.rank
        method = self.method
        print("generating weight matrix...")
        W = self.get_weights_matrix(X)
        U = self.init_rand_matrix(n, rank)
        V = self.init_rand_matrix(rank, m)
        print("starting the iteration...")
        for iter in range(n_iter):
            if self.method == "euclidean":
                U, V, obj_val = self.update_euclidean(X, U, V, W)
            else:
                U, V, obj_val = self.update_divergence(X, U, V, W)
            print(np.sum(U), np.sum(V))
            print("Iteration %d; objective value = %.2f" % (iter, obj_val))
        # set the euclidean length of each col vec in U = 1
        sum_col_U = np.sqrt(np.sum(U**2, axis=0))
        U = U / sum_col_U
        V = V / sum_col_U.reshape((rank, 1))
        return(U, V)
