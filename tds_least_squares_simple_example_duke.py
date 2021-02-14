import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

# https://towardsdatascience.com/total-least-squares-in-comparison-with-ols-and-odr-f050ffc1a86a


def tls(X, y):
    if X.ndim is 1:
        n = 1  # the number of variable of X
        X = X.reshape(len(X), 1)
    else:
        n = np.array(X).shape[1]

    Z = np.vstack((X.T, y)).T
    print(Z)
    U, s, Vt = la.svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy / Vyy  # total least squares soln

    Xtyt = - Z.dot(V[:, n:]).dot(V[:, n:].T)
    Xt = Xtyt[:, :n]  # X error
    y_tls = (X + Xt).dot(a_tls)
    fro_norm = la.norm(Xtyt, 'fro')

    return y_tls, X + Xt, a_tls, fro_norm





N=60
mu=0
sd=2

np.random.seed(0)
ran = np.random.normal(size=N)
error1 = sd**2 * ran + mu
error2 = sd*.5 * ran + mu

lin = np.linspace(-15., 15., num=N)
data = lin + error2
data_true = lin


data = np.array([1,5,7])
y_data = np.array([1,3,8])

# true_func = lambda x, e: .1*x + .1*x**2 + e
x = np.vstack((data, data**2)).T
# y_true = np.array([true_func(d, 0) for d in data_true])
# y_data = np.array([true_func(d, e) for d,e in zip(data, error1)])

y_tls, XplusXt, a_tls, fro_norm = tls(x, y_data)

print(XplusXt.shape)
print(y_tls.shape)
plt.plot(XplusXt[:,0], y_tls,'+')
# plt.plot(data_true,y_true)
plt.plot(data,y_data,'o')
plt.show()
