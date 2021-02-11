import matplotlib.pyplot as plt
import numpy as np

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



# form A
x_regular = np.array(x_center).reshape( (len(x_center), 1))
x_squared = np.square(np.array(x_center)).reshape( (len(x_center), 1) ) # reshape to avoid rank 1 array (26,) shape
b_coeffs = np.ones(shape=x_squared.shape)

b_highest = np.array(y_highest).reshape( (len(y_highest),1) )

A = np.concatenate([x_squared,x_regular,b_coeffs],axis=1)

# params_highest = (ATA)-1  * AT * b
params_highest = np.linalg.pinv(   np.transpose(A).dot(A)   ).dot(  np.transpose(A).dot(y_highest)  )
print("params_highest", params_highest)

x_fit_high = np.linspace(0, np.max(x_regular), 1000)
y_fit_high = params_highest[0] * x_fit_high**2 + params_highest[1] * x_fit_high + params_highest[2]





print(b_highest.shape)

print(x_squared.shape)
print(b_coeffs.shape)

A = np.concatenate([x_squared,b_coeffs],axis=1)
print(A.shape)



# plt.plot(x_squared,y_highest,'o')
plt.plot(x_center, y_highest, '.')
plt.plot(x_center, y_lowest, 'x')
plt.plot(x_fit_high,y_fit_high, '.')

plt.ylabel('y_highest')
plt.show()


