import matplotlib.pyplot as plt
import numpy as np
import random
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




def ordinary_least_squares(highest_row_list, lowest_row_list, center_col_list, img_sizes_list):

    img_height = img_sizes_list[0]
    img_width = img_sizes_list[1]

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

    # b_highest = np.array(y_highest).reshape( (len(y_highest),1) )
    # b_lowest = np.array(y_lowest).reshape( (len(y_lowest),1) )

    A = np.concatenate([x_squared, x_regular, b_coeffs], axis=1)
    print(A.shape)

    # params_highest = (ATA)-1  * AT * b
    params_highest = np.linalg.pinv(np.transpose(A).dot(A)).dot(np.transpose(A).dot(y_highest))
    params_lowest = np.linalg.pinv(np.transpose(A).dot(A)).dot(np.transpose(A).dot(y_lowest))
    # print("params_highest", params_highest)

    # x_fit_high = np.linspace(0, np.max(x_regular), 1000)
    x_fit_high = x_regular
    y_fit_high = params_highest[0] * x_fit_high**2 + params_highest[1] * x_fit_high + params_highest[2]


    # x_fit_low = np.linspace(0, np.max(x_regular), 1000)
    x_fit_low = x_regular
    y_fit_low = params_lowest[0] * x_fit_low**2 + params_lowest[1] * x_fit_low + params_lowest[2]

    y_highest_n = np.array(y_highest).reshape( (len(y_highest), 1) )
    y_lowest_n = np.array(y_lowest).reshape( (len(y_lowest),1) )

    return x_center, y_highest_n, x_center, y_lowest_n, x_fit_high, y_fit_high, x_fit_low, y_fit_low, params_highest, params_lowest

img_height = img_sizes_list[0]
img_width = img_sizes_list[1]
# convert from row column to x y
y_highest = np.array([abs(n-img_height) for n in highest_row_list])
y_lowest = np.array([abs(n-img_height) for n in lowest_row_list])
x_all = np.array(center_col_list)

n = 3 # minimum number of data points required to estimate model parameters
k = 1000 # number of iterations
t = 10 # Threshold value to determine data points that are fit well by model

iterations = 0
maybeInliers = []
bestModel = []
bestErr = -1000

# while(iterations < k):
random_indices_L = random.sample(range(0, len(center_col_list)), 3) # get 3 random not repeating indices
x_random = [center_col_list[i] for i in random_indices_L]
y_random_high = [highest_row_list[i] for i in random_indices_L]

y_random_low = [lowest_row_list[i] for i in random_indices_L]

x_center, y_highest_n, x_center, y_lowest_n, x_fit_high, y_fit_high, x_fit_low, y_fit_low, params_highest, params_lowest = ordinary_least_squares(y_random_high, y_random_low, x_random, img_sizes_list)




# Count close points
print(type(params_highest[0]))
y_fit_all_high = params_highest[0] * np.square(x_all) + params_highest[1] * x_all + params_highest[2]
y_fit_all_low = params_lowest[0] * np.square(x_all) + params_lowest[1] * x_all + params_lowest[2]

errors_high = np.absolute(y_highest - y_fit_all_high)
errors_low = np.absolute(y_lowest - y_fit_all_low)

# errors_high = np.absolute(y_highest_n - y_fit_high)
# errors_low = np.absolute(y_lowest_n - y_fit_low)

print(errors_high.shape)

number_good_high = np.count_nonzero(errors_high <= 10)
number_good_low = np.count_nonzero(errors_low <= 10)
print(number_good_high)
print(number_good_low)

# print("asdf")
# print(np.array(y_highest).shape)
# print(y_fit_high.shape)
# errors_high = np.absolute(y_highest - y_fit_high)
# print(errors_high.shape)

# fit_y_high = np.array(map(lambda x: params_highest[0]*x**2 + params_highest[1]*x + params_highest[2], center_col_list))
# print(fit_y_high)
# print(type(fit_y_high))
# print(type(np.array(y_highest)))
# print(fit_y_high.shape)
# print(np.array(y_highest).shape)
# fit_y_high - np.array(y_highest)
# errors_high = np.absolute(fit_y_high - np.array(y_highest))

# sys.exit()


# x_center, y_highest, x_center, y_lowest, x_fit_high, y_fit_high, x_fit_low, y_fit_low = ordinary_least_squares(highest_row_list, lowest_row_list, center_col_list, img_sizes_list)

plt.plot(x_center, y_highest_n, 'o')
plt.plot(x_center, y_lowest_n, 'o')
plt.plot(x_fit_high, y_fit_high, '.')

plt.plot(x_fit_low, y_fit_low, 'x')

# plt.plot(x_fit_high, errors_high,'x')

plt.ylabel('y_highest')
plt.show()