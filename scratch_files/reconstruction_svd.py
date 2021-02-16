import numpy as np
X = np.random.normal(size=[20,18])
P, D, Q = np.linalg.svd(X, full_matrices=False)
X_a = np.matmul(np.matmul(P, np.diag(D)), Q)
print(np.std(X), np.std(X_a), np.std(X - X_a))

x_ = Q[:,-1]

print(X.dot(x_))