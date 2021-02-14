import matplotlib.pyplot as plt
import numpy as np
import sys

n = 3

center_col_list = [96, 142, 184, 264, 353, 459, 542, 670, 750, 902, 990, 1154, 1286, 1452, 1678, 1872, 2078, 2312, 2530, 2710, 2903, 3189, 3204, 3376]
highest_row_list = [2314, 2174, 2018, 1932, 1806, 1694, 1584, 1484, 1416, 1387, 1244, 1202, 1109, 880, 781, 739, 822, 1121, 1232, 1268, 1533, 1673, 1909, 2130]

center_col_n = np.array(center_col_list).reshape((len(center_col_list),1))
center_col_n_squared = np.square(center_col_n).reshape((len(center_col_list),1))
highest_row_n = np.array(highest_row_list).reshape((len(center_col_list),1))
b_coeffs = np.ones(shape=center_col_n_squared.shape)

X = np.concatenate([b_coeffs,center_col_n,center_col_n_squared],axis=1) # the design matrix
print(X)

Xy_high = np.concatenate([X, highest_row_n],axis=1)
u_high, s_high, vh_high = np.linalg.svd(Xy_high, full_matrices=True)
print(vh_high)

V_high = vh_high.T
print(V_high)

a_tls_high = -V_high[0:n,n] / V_high[n,n]
a_tls_high = a_tls_high.reshape((len(a_tls_high),1))
print(a_tls_high)
print(a_tls_high.shape)

Xtyt_high = -Xy_high.dot(V_high[:,n].reshape(-1,1)).dot( V_high[:,n].reshape(-1,1).T )

Xt_high = Xtyt_high[:,0:n]
print(Xt_high)

y_tls_high = (X+Xt_high).dot(a_tls_high)
print("y_tls_high", y_tls_high)

plt.plot(center_col_list,highest_row_n,'.')
plt.plot((X)[:,1], y_tls_high,'+')

plt.show()

sys.exit()

# print(V_high[:,n].reshape(-1,1).shape)
print(-Xy_high.dot(V_high[:,n]))
print(Xy_high)
print(Xtyt)
