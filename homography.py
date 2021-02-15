import numpy as np



x1 = 5
x2 = 150
x3 = 150
x4 = 5

y1 = 5
y2 = 5
y3 = 150
y4 = 150

xp1 = 100
xp2 = 200
xp3 = 220
xp4 = 100

yp1 = 100
yp2 = 80
yp3 = 80
yp4 = 200

A1 = np.array([-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1])
A2 = np.array([0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1])
A3 = np.array([-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2])
A4 = np.array([0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2])
A5 = np.array([-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3])
A6 = np.array([0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3])
A7 = np.array([-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4])
A8 = np.array([0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4])

A = np.vstack([A1,A2,A3,A4,A5,A6,A7,A8])

# Built in functions
# print(A.shape)
# rhs = np.zeros((A.shape[0],1))
# u, s, vh = np.linalg.svd(A, full_matrices=True)
# print(vh.shape)
# V = vh.T
# x = V[:,-1]
# print(x)
# print(A.dot(x))

