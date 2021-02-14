import matplotlib.pyplot as plt
import numpy as np
import sys

n = 2

center_col_list = [1, 5, 7]
highest_row_list = [1, 3, 8]

center_col_n = np.array(center_col_list).reshape((len(center_col_list),1))
center_col_n_squared = np.square(center_col_n).reshape((len(center_col_list),1))
highest_row_n = np.array(highest_row_list).reshape((len(center_col_list),1))
X = np.concatenate([center_col_n,center_col_n_squared],axis=1) # the design matrix
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
plt.plot((X+Xt_high)[:,0], y_tls_high,'+')

plt.show()

sys.exit()

# print(V_high[:,n].reshape(-1,1).shape)
print(-Xy_high.dot(V_high[:,n]))
print(Xy_high)
print(Xtyt)
