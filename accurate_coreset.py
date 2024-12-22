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

# 1 segment coreset
# Input: a set P of R^{d+1}, where P = {(t_i | p_i)}_{i=1}^n and non-negative weights W
def one_segment(Pset):
    P = Pset.P
    W = Pset.W
    sqrt_W = np.sqrt(W)
    d = Pset.d - 1  # Dimension of the points p_i
    X_unweighted = np.concatenate((np.ones((P.shape[0], 1)), P), axis=1)
    X = (X_unweighted.transpose()*sqrt_W).transpose() # Multiply the i'th row of X_unweighted by the i'th entry of sqrt_W
    U, D, Vt = scipy.linalg.svd(X, full_matrices=False)
    D = np.diag(D)

    # y is the leftmost column of DV^T
    u = np.matmul(D, Vt)[:, 0]

    c = (np.linalg.norm(u) ** 2) / (d + 2)

    w_vec = np.sqrt(c) * np.ones(d + 2)

    # compute a matrix Y such that Yu = w_vec
    Y = align_vectors(u, w_vec)

    # B is the (d+1) rightmost columns of YDV^T/sqrt(c)
    B = (np.matmul(Y, np.matmul(D, Vt)) / np.sqrt(c))[:, 1:]

    # the coreset is the rows of B
    Cset = WeightedSet(B, c*np.ones(B.shape[0]))
    one_segment_coreset_checker(Pset, Cset)

    return Cset, c

def one_segment_coreset_checker(Pset, Cset):
    d = Pset.d - 1
    a = np.random.rand(1, d)
    b = np.random.rand(1, d)

    sum_all_coreset = 0
    sum_all = 0
    for i in range(Pset.n):
        sum_all += Pset.W[0,i]*np.linalg.norm(a + b * Pset.P[i, 0] - Pset.P[i, 1:]) ** 2
    for i in range(Cset.n):
        sum_all_coreset += Cset.W[0,i]*np.linalg.norm(a + b * Cset.P[i, 0] - Cset.P[i, 1:]) ** 2
    if np.abs(sum_all_coreset - sum_all) > small_number:
        print("Bad Coreset, {} - {}".format(sum_all_coreset, sum_all))

# Input: u,v \in R^d
# Find a rotation matrix R such that Ru/||u|| = v/||v||
def align_vectors(u, v):
    d = u.size

    u = np.divide(u, np.linalg.norm(u))
    v = np.divide(v, np.linalg.norm(v))

    u_bot = orthogonal_complement(u)
    v_bot = orthogonal_complement(v)

    # Rotation matrices that align u,v with the x axis
    R_u = np.concatenate((u.reshape(1, d), u_bot.transpose()), axis=0)
    R_v = np.concatenate((v.reshape(1, d), v_bot.transpose()), axis=0)

    # align u and v by first aligning u with the x axis, then aligning the x axis with v
    R_uv = np.matmul(R_v.transpose(), R_u)

    return R_uv


# Given a unit vector x, compute the orthogonal complement of x
def orthogonal_complement(x, threshold=1e-15):
    """Compute orthogonal complement of a matrix
    this works along axis zero, i.e. rank == column rank,
    or number of rows > column rank
    otherwise orthogonal complement is empty
    TODO possibly: use normalize='top' or 'bottom'
    """

    if (abs(np.linalg.norm(x) - 1) > small_number):
        x = np.divide(x, np.linalg.norm(x))

    if (x.shape[0] == x.size):
        x = x.reshape(x.size, 1)

    # x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = scipy.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    return oc

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

    #compute the within cluster sum of squares 
    wcss = kmeans.inertia_
    return wcss



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
    Cset, c = one_segment(Pset)
    print("orginal")
    print(apply_kmeans_on_coreset(Pset, n_clusters=3))

    print("using one segment")
    print(apply_kmeans_on_coreset(Cset, n_clusters=3))


if __name__ == '__main__':
    main()
