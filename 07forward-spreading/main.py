import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""Load data"""
# x: [60k, 28, 28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()

# x: [0-255] -> [0-1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

"""Create dataset"""
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)

"""Making model"""
# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out],[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3  # 10^-3

for epoch in range(30):  # iter db for 10 times
    for step, (x, y) in enumerate(train_db):  # for every batch in db
        # x: [128, 28, 28]
        # y: [128]
        # [b, 28, 28] -> [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        # wrapped in gradient computation
        with tf.GradientTape() as tape:  # tracks only tf.Variable
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # layer 1: [b, 784]@[784, 256] + [256] -> [b, 256]
            h1 = x@w1 + b1  # h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # layer 2: [b, 256]@[256, 128] + [128] -> [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # layer3: [b, 128]@[128, 10] + [10] -> [b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] -> [b, 10]
            y_one_hot = tf.one_hot(y, depth=10)
            # mse = mean ( sum (y-out)^2)
            # loss: [b, 10]
            square_ = tf.square(y_one_hot - out)
            # mean: scalar
            loss = tf.reduce_mean(square_)

        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # adapt changing
        # w1 = w1 - lr * w1_grad
        # assign to oneself
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(step, 'loss: ', float(loss))
