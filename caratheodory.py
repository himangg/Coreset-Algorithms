import numpy as np
from scipy.linalg import null_space

def caratheodory(P, w):
    n = P.shape[0]
    d = 1

    if(n<=d+1):
        return P, w 
    
    A=[]
    for i in range(1,n):
        A.append(P[i]-P[0]) 
    A=np.array(A)

    A = A.reshape(1, -1)

    v=null_space(A)[:, 0]

    v1=0
    for i in range(n-1):
        v1-=v[i]

    v=np.append(v1,v)

    alpha=w[0]/v[0]
    for i in range(n):
        if(v[i]>0):
            alpha=min(alpha,w[i]/v[i])

    u=[]
    S=[]
    for i in range(n):
        if((w[i]-alpha*v[i])>0):
            S.append(P[i])
            u.append(w[i]-alpha*v[i])

    S=np.array(S)
    u=np.array(u)
    if len(P) > d + 1:
        return caratheodory(S, u)
    
    return S, u

P = np.array([1,2,9,6,5]) 
w = np.array([0.25, 0.25, 0.25, 0.15, 0.10]) 
U = np.sqrt(w)

weightedP = np.multiply(P, U[:, None])

x = [3]
v=[]
for i in range(len(weightedP)):
    v.append(weightedP[i]*x)

squaredl2Normofv=np.linalg.norm(v)**2
print(squaredl2Normofv)

S, u = caratheodory(P, w)
# print(S)
# print(u)
u1 = np.sqrt(u)

weightedS = np.multiply(S, u1[:, None])

v=[]
for i in range(len(weightedS)):
    v.append(weightedS[i]*x)

squaredl2Normofv=np.linalg.norm(v)**2
print(squaredl2Normofv)
