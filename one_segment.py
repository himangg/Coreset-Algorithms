import numpy as np
from scipy.linalg import svd

def one_segment(P, w):
    n,d=P.shape
    d-=1

    X = np.zeros((n, d+2))
    for i in range(n):
        t_i = P[i, 0]
        p_i = P[i, 1:]
        sqrt_w = np.sqrt(w[i])
        X[i] = sqrt_w * np.hstack(([1], [t_i], p_i))

    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Sigma_VT = np.diag(Sigma) @ VT
    u = Sigma_VT[:, 0]

    c = (np.linalg.norm(u)**2) / (d+2)
    v = np.full(d+2, np.sqrt(c))
    A = np.eye(d+2)
    A[:, 0] = u 
    Q, _ = np.linalg.qr(A)
    D = np.diag(v / (Q[:, 0] @ u))
    Z = Q @ D


    Z_Sigma_VT = Z @ np.diag(Sigma) @ VT
    Z_Sigma_VT_scaled = Z_Sigma_VT / np.sqrt(c)
    d_plus_2 = Z.shape[0]
    B = Z_Sigma_VT_scaled[:, -d_plus_2+1:]

    C = np.vstack([B.T, u])
    
    u_weights = np.full(C.shape[0], c)
    
    return C, u_weights

P = np.array([
    [0.1, 1.0, 2.0],
    [0.2, 2.0, 3.0],
    [0.3, 3.0, 4.0],
    [0.3, 3.0, 4.0],
    [0.3, 3.0, 4.0],
    [0.3, 3.0, 4.0],
    [0.4, 4.0, 5.0]
])
w = np.array([1.0, 0.8, 0.5, 1.0, 0.7, 0.6, 0.4])

C, u = one_segment(P, w)
print("Coreset C:\n", C)
print("Weights u:\n", u)
