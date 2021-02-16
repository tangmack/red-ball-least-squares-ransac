import numpy as np
import scipy
# Following simple example from https://www.youtube.com/watch?v=cOUTpqlX-Xs

# C = np.array([[5, 51, 17], [-1, 7, 20]])
C = np.array([[1, 3, -4],[2,-5, 0],[-2,-6, 8]])

eigenValues, eigenVectors = np.linalg.eig(C.T.dot(C)) # compute eigenvalues of CTC

idx = eigenValues.argsort()[::-1]
d = eigenValues[idx]
v = eigenVectors[:,idx]

print(d)
print(v)

V = v
Sigma = np.sqrt(np.abs(np.diag(d)))
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
print("custom final check: ", FinalCheck)

############# Now that we have SVD, use it to find the pseudoinverse ############
# The solution to Ax = 0 is given by the last column of the matrix V

x = V[:,-1].reshape(-1,1)
print(V)
# print(x)
# print(C.dot(x))

print("built in section")
u, s, vh = np.linalg.svd(C, full_matrices=True)

x_b = vh.T[:,-1].reshape(-1,1)
print(vh)
# print(x_b)
# print(C.dot(x_b))
# X_a = u @ np.diag(s) @ vh

# print(X_a)

print(C.dot(x_b))

# print(np.std(C), np.std(X_a), np.std(C - X_a))
# print('Is X close to X_a?', np.isclose(C, X_a).all())

print(" ")
print(V.T)
print(" ")
print(vh)

print("x built in: ", x_b)

print(" ...  ")

print(u)
# DD = np.concatenate( (np.diag(s), np.array([[0],[0]]) ), axis=1   )
# print(   np.concatenate( (np.diag(s), np.array([[0],[0]]) ), axis=1   )   )

print(vh)

# print(u.dot(DD.dot(vh)))

print("custom ....")
print(U)
print(Sigma)
print(V.T)

print("V custom: ", V)

print("x custom: ", x)
