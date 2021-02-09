import matplotlib.pyplot as plt

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
x_center = [abs(n-img_width) for n in center_col_list]
print(y_highest)

plt.plot(x_center, y_highest, '.')
plt.plot(x_center, y_lowest, 'x')


plt.ylabel('y_highest')
plt.show()


