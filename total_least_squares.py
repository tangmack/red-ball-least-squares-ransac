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
X_low = np.concatenate([b_coeffs,x_regular,x_squared],axis=1) # the design matrix


y_high_n = np.array(y_highest).reshape( ( len(x_center),1 ) )
y_low_n = np.array(y_lowest).reshape( ( len(x_center),1 ) )

Xy_high = np.concatenate([X_high, y_high_n],axis=1)
Xy_low = np.concatenate([X_low, y_low_n],axis=1)

u_high, s_high, vh_high = np.linalg.svd(Xy_high, full_matrices=True)
V_high = vh_high.T

u_low, s_low, vh_low = np.linalg.svd(Xy_low, full_matrices=True)
V_low = vh_low.T

##### For High #####################################################
n = 3 # number of parameters
a_tls_high = -V_high[0:n, n] / V_high[n, n]
a_tls_high = a_tls_high.reshape((len(a_tls_high), 1))
Xtyt_high = -Xy_high.dot(V_high[:,n].reshape(-1,1)).dot( V_high[:,n].reshape(-1,1).T )

Xt_high = Xtyt_high[:,0:n]
print(Xt_high.shape)
print(X_high.shape)

y_tls_high = (X_high+Xt_high).dot(a_tls_high)

##### For Low #####################################################
n = 3 # number of parameters
a_tls_low = -V_low[0:n, n] / V_low[n, n]
a_tls_low = a_tls_low.reshape((len(a_tls_low), 1))
Xtyt_low = -Xy_low.dot(V_low[:,n].reshape(-1,1)).dot( V_low[:,n].reshape(-1,1).T )

Xt_low = Xtyt_low[:,0:n]
print(Xt_low.shape)
print(X_low.shape)

y_tls_low = (X_low+Xt_low).dot(a_tls_low)

##### Plot #####################################################

plt.plot(x_regular,y_high_n,'.')
plt.plot((X_high+Xt_high)[:,1], y_tls_high,'+')

plt.plot(x_regular,y_low_n,'.')
plt.plot((X_low+Xt_low)[:,1], y_tls_low,'x')

# Plot continuous lines from TLS
x_fit_high = np.linspace(0, np.max(x_regular), 1000)
# y_fit_high = a_tls_high[0] * x_fit_high**2 + a_tls_high[1] * x_fit_high + a_tls_high[2]
y_fit_high = a_tls_high[0] + a_tls_high[1] * x_fit_high + a_tls_high[2] * x_fit_high**2
plt.plot(x_fit_high,y_fit_high,'o')

x_fit_low = np.linspace(0, np.max(x_regular), 1000)
y_fit_low = a_tls_low[0] + a_tls_low[1] * x_fit_low + a_tls_low[2] * x_fit_low**2
plt.plot(x_fit_low,y_fit_low,'.')

plt.show()
sys.exit()






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
