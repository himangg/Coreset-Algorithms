"""*****************************************************************************************
MIT License
Copyright (c) 2019 Ibrahim Jubran, Alaa Maalouf, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


################################### NOTES ###########################################
# - Please cite our paper when using the code:
#                "Accurate Coresets"
#    Ibrahim Jubran and Alaa Maalouf and Dan Feldman
#
# - Code for other coresets, both accurate and eps-coresets, will be published soon.
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import time
from helper_functions import Fast_Caratheodory, train_model, get_new_clf, test_model
from sklearn import linear_model
from sklearn.decomposition import PCA


class WeightedSet:

    def __init__(self, P, W, Y=None):
        # P is of size nXd
        # W is of size 1Xn
        # Y is of size nXk
        ##todo add checker

        self.P = np.array(P)
        if (P.ndim == 1):
            self.P = self.P.reshape(-1, 1)
        self.n = self.P.shape[0]
        self.d = self.P.shape[1]
        self.Y = Y
        self.W = np.array(W).reshape(1, -1)
        self.dtype = P.dtype

        if (self.Y is not None) and (P.shape[0] != Y.shape[0]):
            self.Y = self.Y.reshape(self.n, -1)

        self.sum_W = np.sum(self.W);  # print (self.W.shape)
        self.weighted_sum = self.P.T.dot(self.W.T)

    # p must be a 1 dimensional nparray (p.shape = (d,) )
    # w is a number
    def add_point(self, p, w, y):
        self.P = np.append(self.P, [p], axis=0)
        self.W = np.append(self.W, w)
        if not self.Y is None: self.Y = np.append(self.Y, y)
        self.n = self.n + 1
        self.sum_W = self.sum_W + w
        self.weighted_sum = self.weighted_sum + w * p

def get_normalized_weighted_set(Pset):
    W_normalized = Pset.W / Pset.sum_W
    return WeightedSet(Pset.P, W_normalized)

# 1 mean coreset 3
# Input: a weighted set in R^d. The sum of weights is not necesarrily 1. Therefore, we first divide the input weights by
# their sum, compute the subset coreset of bounded weights using Caratheodory's theorem, then multiply the obtained
# weights by the original sum of weights.
def one_mean_3(Pset):
    pre_norm_sum_W = Pset.sum_W
    pre_norm_weighted_sum = Pset.weighted_sum
    Pset = get_normalized_weighted_set(Pset)

    # Add 2 more dimensions to each point: p -> (p, ||p||, 1)
    Q = Pset.P
    Q_norms = np.linalg.norm(Q, axis=1)
    Q = np.concatenate((Q, np.power(Q_norms.reshape(Q_norms.shape[0], 1),2), np.ones((Q.shape[0], 1))), axis=1)

    # Create another weighted set with Q and the same weights of Pset
    Qset = WeightedSet(Q, Pset.W)

    # Compute the weighted sum of Qset as a convex combination of at most d+3 points from Qset. d is the dimension of P
    W_cara  = Fast_Caratheodory(Qset.P, Qset.W, Qset.d + 1)
    C_idx = np.nonzero(W_cara)
    P_cara = Qset.P[C_idx]
    W_cara = W_cara[C_idx]

    # testing
    Tset = WeightedSet(P_cara, W_cara)
    if (np.linalg.norm(Tset.weighted_sum - Qset.weighted_sum) > small_number):
        print("Bad coreset!!")
        return Pset

    # Output coreset: the points of P that correspond to the points of Q that were chosen in the function updated_cara
    Cset = WeightedSet(Pset.P[C_idx], W_cara * pre_norm_sum_W)

    # Check that the weighted coreset points = weighted input points
    if (np.linalg.norm(Cset.weighted_sum - pre_norm_weighted_sum) > small_number):
        print("Bad result - wrong weighted sum!!")
        return Pset

    # check that the sum of weights of the coreset = sum of weights of the input points
    if (abs(Cset.sum_W - pre_norm_sum_W) > small_number):
        print("Bad result - wrong sum of weights")
        return Pset

    return Cset

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
import numpy as np

def apply_kmeans_on_coreset(Cset, n_clusters=3):
    """
    Applies KMeans clustering to the weighted coreset.

    Parameters:
    - Cset: WeightedSet, the coreset with points in Cset.P and weights in Cset.W
    - n_clusters: int, the number of clusters to form

    Returns:
    - centroids: array, coordinates of cluster centers
    - labels: array, label of each point
    """
    # Since sklearn's KMeans does not directly support weighted points,
    # we will replicate points according to their weights for a simple approximation.
    # Note: This is a heuristic and may not always yield perfect results.

    # Calculate integer weights proportional to Cset.W
    if np.min(Cset.W) < 1:
        weight_factor = 1 / np.min(Cset.W)
    else:
        weight_factor = 1
    int_weights = np.ceil(Cset.W * weight_factor).astype(int).flatten()

    # Replicate points based on their weights
    weighted_points = np.repeat(Cset.P, int_weights, axis=0)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(weighted_points)

    # Extract results
    centroids = kmeans.cluster_centers_
    labels = kmeans.predict(Cset.P)  # predicting labels of original coreset points

    return centroids


def main():
    global small_number
    small_number = 0.000001

    iris = load_iris()
    X = iris.data  # 150 samples and 4 features
    W = np.ones(X.shape[0])  # Assign uniform weights

    # Prepare the dataset according to the expected format
    # If Y is not used, set it to None or adjust accordingly if it's meant to hold specific data.
    Pset = WeightedSet(X, W, None)  # Passing None for Y since it doesn't seem to be used.

    # Assuming one_segment is expected to be used, let's correct that function call as well.
    # Ensure one_segment function and any other usage correctly handles this setup.
    Cset = one_mean_3(Pset)
    #print the mean of the pset
    # print(np.mean(Cset.P,axis=0))
    # print(Pset.P)
    print(Cset.W.shape)
    print("orginal")
    print(apply_kmeans_on_coreset(Pset, n_clusters=1))

    print("using one segment")
    print(apply_kmeans_on_coreset(Cset, n_clusters=1))


if __name__ == '__main__':
    main()
