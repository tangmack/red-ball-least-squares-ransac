import numpy as np
# Following simple example from https://www.youtube.com/watch?v=cOUTpqlX-Xs

C = np.array([[5, 5], [-1, 7]])

d, v = np.linalg.eig(C.dot(C.T)) # compute eigenvalues of CTC

print(d)
print(v)

V = v
Sigma = np.sqrt(np.diag(d))
print(Sigma)

# want C = U Sigma V.T


# We have Sigma and V, now Find U
# C V = U Sigma

C_times_V = C.dot(V)

print(C_times_V)

U = C_times_V
print(U)
for i in range(0,V.shape[1]): # for every column of U (every eigenvalue)
    U[:,i] = U[:,i] / Sigma[i,i]

print(U)

FinalCheck = U.dot(Sigma.dot(V.T))
print(FinalCheck)