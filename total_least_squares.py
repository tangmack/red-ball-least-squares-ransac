import matplotlib.pyplot as plt
import numpy as np
import sys


with open('highest_row_list.txt', 'r') as f:
    highest_row_strings = [line.rstrip('\n') for line in f]
    highest_row_list = [int(i) for i in highest_row_strings]

with open('lowest_row_list.txt', 'r') as f:
    lowest_row_strings = [line.rstrip('\n') for line in f]
    lowest_row_list = [int(i) for i in lowest_row_strings]

with open('highest_col_list.txt', 'r') as f:
    center_col_strings = [line.rstrip('\n') for line in f]
    center_col_list = [int(i) for i in center_col_strings]

with open('img_size.txt', 'r') as f:
    img_sizes_strings = [line.rstrip('\n') for line in f]
    img_sizes_list = [int(i) for i in img_sizes_strings]

print(highest_row_list)
print(lowest_row_list)
print(center_col_list)

print(highest_row_list)

print(len(highest_row_list))


img_height = img_sizes_list[0]
img_width = img_sizes_list[1]

print(img_height, img_width)

print(img_sizes_strings)

# convert from row column to x y
y_highest = [abs(n-img_height) for n in highest_row_list]
y_lowest = [abs(n-img_height) for n in lowest_row_list]
x_center = center_col_list
# x_center = [abs(n-img_width) for n in center_col_list]
print(y_highest)


# Form [X y]
x_squared = np.square(np.array(x_center)).reshape( (len(x_center), 1) ) # reshape to avoid rank 1 array (26,) shape
x_regular = np.array(x_center).reshape( (len(x_center), 1))
b_coeffs = np.ones(shape=x_squared.shape)

X_high = np.concatenate([b_coeffs,x_regular,x_squared],axis=1) # the design matrix

y_high_n = np.array(y_highest).reshape( ( len(x_center),1 ) )
y_low_n = np.array(y_lowest).reshape( ( len(x_center),1 ) )

Xy_high = np.concatenate([X_high, y_high_n],axis=1)
# Xy_high = np.concatenate([x_squared, x_regular, b_coeffs, y_high_n], axis=1)
# Xy_low = np.concatenate([x_squared, x_regular, b_coeffs, y_low_n], axis=1)
#
# print(Xy_high.shape)
#
# u_high, s_high, vh_high = np.linalg.svd(Xy_high, full_matrices=True)
# u_low, s_low, vh_low = np.linalg.svd(Xy_low, full_matrices=True)
# print(vh_high.shape) # shape should be
# print(vh_high)

u_high, s_high, vh_high = np.linalg.svd(Xy_high, full_matrices=True)
V_high = vh_high.T


# Vhigh = vh_high.T
# Vlow = vh_low.T


##### For High #####################################################
# n = 3 # number of parameters
# m = len(x_center) # number of points
# v_pq_high = Vhigh[0:n,n] # the first n elements of the (n+1)th column of vh
# v_qq_high = Vhigh[n,n] # the n+1 element of the n+1 column of vh
#
# a_tls_high = -v_pq_high / v_qq_high
# print("a_tls_high", a_tls_high)
#
# v_pq_qq_high = Vhigh[:,n].reshape(n+1,1) # last column (n+1 column)
n = 3 # number of parameters
a_tls_high = -V_high[0:n, n] / V_high[n, n]
a_tls_high = a_tls_high.reshape((len(a_tls_high), 1))
Xtyt_high = -Xy_high.dot(V_high[:,n].reshape(-1,1)).dot( V_high[:,n].reshape(-1,1).T )

Xt_high = Xtyt_high[:,0:n]
print(Xt_high.shape)
print(X_high.shape)

y_tls_high = (X_high+Xt_high).dot(a_tls_high)

# plt.plot(x_regular,y_high_n,'.')
# plt.plot((X_high+Xt_high)[:,1], y_tls_high,'+')

plt.plot(x_regular,Xt_high[:,1])

plt.show()

sys.exit()


# Xtilda_ytilda_high = -Xy_high.dot(v_pq_qq_high).dot(v_pq_qq_high.T)
#
# # X_high = Xy_high[:,0:n] # all but last column
# X_tilda_high = Xtilda_ytilda_high[:,0:n]
#
# y_tls_high = (X_high + X_tilda_high).dot(a_tls_high)
# print(y_tls_high.shape)

##### For Low #####################################################
v_pq_low = Vlow[0:n,n] # the first n elements of the (n+1)th column of vh
v_qq_low = Vlow[n,n] # the n+1 element of the n+1 column of vh

a_tls_low = -v_pq_low / v_qq_low

v_pq_qq_low = Vlow[:,n].reshape(n+1,1) # last column (n+1 column)


Xtilda_ytilda_low = -Xy_low.dot(v_pq_qq_low).dot(v_pq_qq_low.T)

X_low = Xy_low[:,0:n] # all but last column
X_tilda_low = Xtilda_ytilda_low[:,0:n]

print("Xtilda low: ", Xtilda_ytilda_low)

y_tls_low = (X_low + X_tilda_low).dot(a_tls_low)
print(y_tls_low.shape)

##### Plot #####################################################
plt.plot(x_center, y_high_n, '.')
# plt.plot(x_center, y_low_n, 'x')
#
# plt.plot(x_center,y_tls_high, '.'
plt.plot(x_center, y_tls_high.reshape(m,1), '.')

# plt.plot(x_center,y_tls_low, '.')

# plt.plot(x_center, tls(X_high,y_high_n)[0])

plt.ylabel('y_highest')
plt.show()


print(y_tls_high.reshape(m,1))
print(y_high_n.shape)


mydiff = np.subtract(y_tls_high.reshape(m,1), y_high_n)
print(mydiff.shape)
print(mydiff)
