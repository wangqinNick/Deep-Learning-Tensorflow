import numpy as np


def compute_error_for_line_given_points(b, w, points):
    """
    compute the loss function of a linear regression
    :param b: intersection
    :param w: gradient
    :param points: array of points [x_, y_]
    :return: the average loss of the regression model
    """
    total_loss = 0
    for i in range(len(points)):
        x = points[i, 0]  # points[i, 0] = points[i][0]
        y = points[i, 1]  # points[i, 1] = points[i][1]
        loss = (w * x + b - y) ** 2
        total_loss += loss
    return total_loss / float(len(points))


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += (2/N) * (w_current * x + b_current - y)
        w_gradient += (2/N) * (w_current * x + b_current - y) * x

    b_next = b_current - learning_rate * b_gradient
    w_next = w_current - learning_rate * w_gradient
    return b_next, w_next


def gradient_descent_runner(points, b_starting, w_starting, learning_rate, num_iterations):
    """
    wrap the step_gradient function in a for loop to continuously compute the b and w
    :param points: an array of x_ and y_
    :param b_starting:
    :param w_starting:
    :param learning_rate:
    :param num_iterations:
    :return: mature b and w
    """
    b = b_starting
    w = w_starting
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learning_rate)
    return b, w


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y_-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 100000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()

