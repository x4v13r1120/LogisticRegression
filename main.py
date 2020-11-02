import numpy as np
import matplotlib.pyplot as plt


# draws a line on a plot given two points
def draw(x1, x2):
    ln = plt.plot(x1, x2)


# sigmoid function for error function
def sigmoid(score):
    return 1 / (1 + np.exp(-score))


# cross entropy formula -sigma yln(p) + (1-y)(ln(1-p))
def calculate_error(line_parameters, points, y):
    n = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1 / n) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


# gradient descent formula 1/n sigma_i x_i(a(x)-y)
def gradient_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        gradient = points.T * (p - y) * (alpha / n)
        line_parameters = line_parameters - gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + (x1 * (-w1 / w2))
    draw(x1, x2)


# number of points & seed placer holder in case i want to test same random numbers
n_pts = 100
np.random.seed(0)

# perceptron theory along with x1 and x2 node bias is an input as well ,as 1
bias = np.ones(n_pts)

# sets the two regions that can be classified with random sample numbers and for vertical and horizontal pair
# we transpose. standard dev is 2 . center of point distribution is first number.
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T

all_points = np.vstack((top_region, bottom_region))

# creates a matrix using the weights and bias and for vertical and horizontal pair we transpose.
line_parameters = np.matrix([np.zeros(3)]).T

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)
# display multiple plots on same figure and specific figure size which is 4 inches wide and 4 inches wide.
# splits into two values first blank second ax
_, ax = plt.subplots(figsize=(4, 4))

# first two arguments are horizontal and vertical points third is color of points
# accessed from our regions top and bottom and index every single row in first and second list
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

gradient_descent(line_parameters, all_points, y, 0.06)

# draws our  plot
plt.show()
