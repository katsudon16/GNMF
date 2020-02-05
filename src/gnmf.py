import numpy as np
import time

class Gnmf(object):
    """
    Graph regularized NMF
    """
    def __init__(self, X, rank=10, p=3, W=None, lmbda=100, method="euclidean"):
        """
        - X     : the original matrix
        - rank  : NMF rank
        - p     : # closest neighbors to be taken into account in the weight matrix
        - W     : the weight matrix - must be symmetric; p will be ignored if W is provided
        - method: "euclidean" or "divergence"
        """
        self.X = X
        self.rank = rank
        self.method = method
        self.lmbda = lmbda
        self.W = W
        self.p = p
        if self.W is None:
            self.W = self.calc_weights_matrix(self.X)
        elif not self.is_matrix_symmetric(self.W):
            raise ValueError("The provided weight matrix should be symmetric")

    def init_rand_matrix(self, nrow, ncol, seed=None):
        """
        Initialize matrix (random) given # rows and # cols
        """
        if not seed:
            seed = np.random.randint(1000)
        np.random.seed(seed)
        return(np.random.dirichlet(np.ones(nrow), size=ncol).T)

    def calc_weights_matrix(self, X):
        """
        Generate weights matrix by p-nearest neighbors + dot-product weighting
        Complexity: O(N^2 * M)
        """
        print("generating weight matrix...")
        time_cp1 = time.time()
        _, m = X.shape
        p = self.p
        # initialize distance matrix and W matrix
        dist_matrix = np.full((m, m), np.inf)
        W = np.zeros((m, m))
        #TODO*: implementation improvement? remove for loops
        # src: https://ljvmiranda921.github.io/notebook/2017/02/09/k-nearest-neighbors/
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
        time_cp2 = time.time()
        print("total time: ", time_cp2 - time_cp1)
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
        """
        Ensure all values in a numpy array > 0
        """
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
        # update V
        #TODO*: improve using iterative algorithm CG
        V = V * (U.T @ np.divide(X, U @ V))
        U_row_sum = np.sum(U, axis=0).reshape((k, 1))
        for i in range(k):
            V[i] = V[i] @ np.linalg.pinv(U_row_sum[i] * np.identity(m) + lmbda * L)
        # update U
        V_col_sum = np.sum(V, axis=1).reshape((1, k))
        U = U * np.divide(np.divide(X, U @ V) @ V.T, V_col_sum)
        # calc obj_val
        X_temp = U @ V
        obj_val = np.sum(X * np.log(
        self.np_pos(np.divide(self.np_pos(X), self.np_pos(X_temp)), add_eps=True)
        ) - X + X_temp)
        return(U, V, obj_val)

    def is_matrix_symmetric(self, M, rtol=1e-05, atol=1e-08):
        """
        Check if the given matrix M is symmetric
        """
        return(np.allclose(M, M.T, rtol=rtol, atol=atol))

    def factorize(self, n_iter=100, return_obj_values=False):
        """
        Factorize matrix X into W and H given rank using multiplicative method
        params:
        - n_iter           : the number of iterations
        - return_obj_values: enable returning the list of produced objective function values as the third tuple element
        Returns a tuple (U, V) or (U, V, obj_values if return_obj_values is True)
        """
        X = self.X
        n, m = X.shape
        rank = self.rank
        method = self.method
        W = self.W

        # initialize U & V
        U = self.init_rand_matrix(n, rank)
        V = self.init_rand_matrix(rank, m)

        # print("running GNMF given matrix %dx%d, rank %d, %d neighbors, lambda %d, %d iterations" % (n, m, rank, self.p, self.lmbda, n_iter))
        time_cp1 = time.time()
        obj_vals = [] # a list of the produced objective function values
        curr_obj_val = float("inf")
        for iter in range(n_iter):
            # if iter % 20 == 0:
            #     print("Completed %d iterations..." % iter)

            U, V, obj_val = (self.update_euclidean(X, U, V, W)
                            if self.method == "euclidean"
                            else self.update_divergence(X, U, V, W))

            # check if the objective function value is decreasing
            if curr_obj_val < obj_val:
                print("The objective function value is not decreasing!! :'(")
                break
            curr_obj_val = obj_val
            obj_vals.append(obj_val)

        time_cp2 = time.time()
        print("total duration: %d; avg duration/iter: %.2f" % (time_cp2 - time_cp1, (time_cp2 - time_cp1) / n_iter))

        # set the euclidean length of each col vec in U = 1
        sum_col_U = np.sqrt(np.sum(U**2, axis=0))
        U = U / sum_col_U
        V = V / sum_col_U.reshape((rank, 1))

        if return_obj_values:
            return(U, V, obj_vals)
        return(U, V)

    def get_weight_matrix(self):
        """
        Retrieve the weight matrix W
        """
        return self.W
