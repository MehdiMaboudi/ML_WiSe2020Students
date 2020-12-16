import numpy as np

class KNearestNeighbor():
    """ a kNN classifier with L2 distance 
        Our conventions:
            N: Number of trainig samples
            D: Dimensionality (Number of features)
    """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier.
        For k-nearest neighbors this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (N, D) containing the training data
          consisting of N samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for query data using this classifier.

        Inputs:
        - X: A numpy array of shape (Nq, D) containing query data consisting
             of Nq samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and query points.

        Returns:
        - y: A numpy array of shape (Nq,) containing predicted labels for the
          query data, where y[i] is the predicted label for the query sample X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each query point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        query data.

        Inputs:
        - X: A numpy array of shape (Nq, D) containing query data.

        Returns:
        - dists: A numpy array of shape (Nq, N) where dists[i, j]
          is the Euclidean distance between the ith query point and the jth training
          point.
        """
        num_query = X.shape[0]  #Number of query samples: Nq
        num_train = self.X_train.shape[0]#Number of training samples: N
        dists = np.zeros((num_query, num_train))
        for i in range(num_query):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith query point and the jth   #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                dists[i,j]=np.sum((X[i,:]-self.X_train[j,:])**2)

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each query point in X and each training point
        in self.X_train using a single loop over the query data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_query = X.shape[0]  #Number of query samples: Nq
        num_train = self.X_train.shape[0]#Number of training samples: N
        dists = np.zeros((num_query, num_train))
        for i in range(num_query):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith query point and all training#
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            dists[i, :] = np.sum((self.X_train - X[i, :]) ** 2, axis=1)
        
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each query point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_query = X.shape[0]  #Number of query samples: Nq
        num_train = self.X_train.shape[0]#Number of training samples: N
        dists = np.zeros((num_query, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all query points and all training     #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
        X_sums = np.sum(X ** 2, axis=1, keepdims=True)
        X_train_sums = np.sum(self.X_train ** 2, axis=1)
        X_sumsT = -2.0 * X.dot(self.X_train.T)
        
        
        dists = X_sums + X_sumsT + X_train_sums
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between query points and training points,
        predict a label for each query point.

        Inputs:
        - dists: A numpy array of shape (num_query, num_train) where dists[i, j]
          gives the distance betwen the ith query point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_query,) containing predicted labels for the
          query data, where y[i] is the predicted label for the query point X[i].
        """
        num_query = dists.shape[0]#Number of query samples: Nq
        y_pred = np.zeros(num_query)
        for i in range(num_query):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith query point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # query point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            a = 0
            if k<= dists.shape[1]:
                while a <k:
                    position_closest_neighbor = np.argsort(dists, axis=1)[i,a:a+1]
                    closest_y.append(int(self.y_train[position_closest_neighbor]))
                    a= a+1
            else:
                print('k out of range for k = ',k)
                break
                
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            y_pred[i]=int(max(closest_y,key=closest_y.count))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
