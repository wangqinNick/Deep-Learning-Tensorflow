import tensorflow as tf


# define function
def himmelblau(x):
    """
    f(x_, y_) =(x_^2 + y_ - 11)^2 + (x_ + y_^2 - 7)^2
    :param x: a couple (x_, y_)
    :return: f(X, y_)
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = tf.constant([4., 0.])

for step in range(200):
    # initialize x_

    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y, [x])[0]

    x -= 0.01 * grads
    if step % 20 == 0:
        print('step{}: x_ = {}, f(x_) = {}'
              .format(step, x.numpy(), y.numpy()))
