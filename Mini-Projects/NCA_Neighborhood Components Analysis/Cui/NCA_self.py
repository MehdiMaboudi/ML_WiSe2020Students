import numpy as np


# 利用Python实现NCA,Neighbor Component Analysis
# 逐步求解梯度，涉及到求和利用两层for循环来解决

class NCA():
    def __init__(self, low_dims=2, learning_rate=0.3, max_steps=500, random_state=0, init_style="normal", code_style = "#2"):
        '''
        init function
        @params low_dims : the dimension of transformed data
        @params learning_rate : default 0.01
        @params max_steps : the max steps of gradient descent, default 500
        '''
        self.low_dims = low_dims
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.init_style = init_style
        self.random_state = random_state
        self.code_style = code_style

    def fit(self, X, Y):
        '''
        train on X and Y, supervised, to learn a matrix A
        maximize \sum_i \sum_{j \in C_i} frac{exp(-||Ax_i-Ax_j||^2)}{\sum_{k neq i} exp(-||Ax_i-Ax_k||^2)}
        @params X : 2-d numpy.array
        @params Y : 1-d numpy.array
        '''
        (n, d) = X.shape
        self.n_samples = n
        self.high_dims = d
        # parametric matrix
        self.A = self.get_random_params(shape=(self.high_dims, self.low_dims))

        # training using gradient descent
        s = 0
        target = 0
        while s < self.max_steps:
            # if s % 2 == 0 and s > 1:
            #     print("Step {}, target = {}...".format(s, target))

            pij_mat, pi_row = self.M_distance(X, Y, self.A)

            # target
            target = np.sum(pi_row)

            if self.code_style == "loop":
                # gradients for-loop
                gradients = np.zeros((self.high_dims, self.high_dims))
                # for i
                for i in range(self.n_samples):
                    k_sum = np.zeros((self.high_dims, self.high_dims))
                    k_same_sum = np.zeros((self.high_dims, self.high_dims))
                    # for k
                    for k in range(self.n_samples):
                        out_prod = np.outer(X[i] - X[k], X[i] - X[k])
                        k_sum += pij_mat[i][k] * out_prod
                        if Y[k] == Y[i]:
                            k_same_sum += pij_mat[i][k] * out_prod
                    gradients += pi_row[i] * k_sum - k_same_sum
                gradients = 2 * np.dot(gradients, self.A)


            # # gradient #1 matrix
            # part_gradients = np.zeros((self.high_dims, self.high_dims))
            # for i in range(self.n_samples):
            #     xik = X[i] - X
            #     prod_xik = xik[:, :, None] * xik[:, None, :]
            #     pij_prod_xik = pij_mat[i][:, None, None] * prod_xik
            #     first_part = pi_row[i] * np.sum(pij_prod_xik, axis=0)
            #     second_part = np.sum(pij_prod_xik[Y == Y[i], :, :], axis=0)
            #     part_gradients += first_part - second_part
            # gradients = 2 * np.dot(part_gradients, self.A)

            if self.code_style == "#2":
                # gradients #2
                part_gradients = np.zeros((self.high_dims, self.high_dims))
                for i in range(self.n_samples):
                    xik = X[i] - X
                    prod_xik = xik[:, :, None] * xik[:, None, :]
                    pij_prod_xik = pij_mat[i][:, None, None] * prod_xik
                    first_part = pi_row[i] * np.sum(pij_prod_xik, axis=0)
                    second_part = np.sum(pij_prod_xik[Y == Y[i], :, :], axis=0)
                    part_gradients += first_part - second_part
                gradients = 2 * (part_gradients @ self.A)

            # update
            self.A += self.learning_rate * gradients

            # step++
            s += 1

    def transform(self, X):
        '''
        transform X from high dimension space to low dimension space
        '''
        low_X = X @ self.A
        return low_X

    def fit_transform(self, X, Y):
        '''
        train on X
        and then
        transform X from high dimension space to low dimension space
        '''
        self.fit(X, Y)
        low_X = self.transform(X)
        return low_X

    def get_random_params(self, shape):
        '''
        get parameter init values
        @params shape : tuple
        '''
        rng = np.random.RandomState(self.random_state)
        if self.init_style == "normal":
            return 0.1 * rng.standard_normal(size=shape)
        elif self.init_style == "uniform":
            return rng.uniform(size=shape)
        else:
            print("No such parameter init style!")
            raise Exception

    def M_distance(self, X, Y, A):

        # to low dimension
        low_X = np.dot(X, A)

        # distance matrix
        sum_row = np.sum(low_X ** 2, axis=1)
        # xxt = np.dot(low_X, low_X.transpose())
        # dist_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * xxt
        dist_mat = np.sum((low_X[None, :, :] - low_X[:, None, :]) ** 2, axis=2)

        # distance to probability
        exp_neg_dist = np.exp(0.0 - dist_mat)
        exp_neg_dist = exp_neg_dist - np.diag(np.diag(exp_neg_dist))
        pij_mat = exp_neg_dist / np.sum(exp_neg_dist, axis=1).reshape((-1, 1))

        # pi = \sum_{j \in C_i} p_{ij}
        pi_row = np.array([np.sum(pij_mat[i][Y == Y[i]]) for i in range(self.n_samples)])
        #####
        return pij_mat, pi_row
