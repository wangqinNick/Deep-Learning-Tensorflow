import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x_, y_):
    x_ = tf.cast(x_, dtype=tf.float32) / 255.
    y_ = tf.cast(y_, dtype=tf.int32)
    return x_, y_


def mnist_dataset():
    # import data
    (x, y), (x_val, y_val) = keras.datasets.cifar10.load_data()

    ds_ = tf.data.Dataset.from_tensor_slices((x, y))
    ds_ = ds_.map(preprocess)  # x: [50k, 32, 32, 3] y:[50k, 1, 10]

    ds_ = ds_.shuffle(60000)
    ds_ = ds_.batch(100)

    ds_val_ = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val_ = ds_val_.map(preprocess)
    ds_val_ = ds_val_.shuffle(10000)
    ds_val_ = ds_val_.batch(100)

    return ds_, ds_val_


def my_nn(train_db_):
    w1 = tf.Variable(tf.random.truncated_normal([1024 * 3, 512 * 3], stddev=0.1))
    b1 = tf.Variable(tf.zeros([512 * 3]))
    w2 = tf.Variable(tf.random.truncated_normal([512 * 3, 256 * 3], stddev=0.1))
    b2 = tf.Variable(tf.zeros([256 * 3]))
    w3 = tf.Variable(tf.random.truncated_normal([256 * 3, 128 * 3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([128 * 3]))
    w4 = tf.Variable(tf.random.truncated_normal([128 * 3, 10 * 3], stddev=0.1))
    b4 = tf.Variable(tf.zeros([10 * 3]))
    w5 = tf.Variable(tf.random.truncated_normal([10 * 3, 10], stddev=0.1))
    b5 = tf.Variable(tf.zeros([10]))
    lr = 1e-3

    for epoch in range(100):  # iterate db for 10
        for step, (x, y) in enumerate(train_db_):
            # x: [100, 32, 32, 3] (100 pics in one batch)
            # y: [100, 10]

            # flat x: [100, 32, 32, 3] -> [100, 32*32, 3]
            x = tf.reshape(x, [100, -1])
            with tf.GradientTape() as tape:  # tf.Variable
                # x: [b, 32*32]
                # h1 = x@w1 + b1
                # [b, 1024]@[1024, 512] + [512] => [b, 512] + [512] => [b, 512] + [b, 512]
                h1 = x @ w1 + b1
                # wrap in relu
                h1 = tf.nn.relu(h1)

                # [b, 512] -> [b, 256]
                h2 = h1 @ w2 + b2
                # wrap in relu
                h2 = tf.nn.relu(h2)

                # [b, 256] -> [b, 128]
                h3 = h2 @ w3 + b3
                # wrap in relu
                h3 = tf.nn.relu(h3)

                # [b, 128] -> [b, 10]
                h4 = h3 @ w4 + b4
                h4 = tf.nn.relu(h4)

                out = h4 @ w5 + b5

                # convert y to one-hot value
                y_onehot = tf.one_hot(y, depth=10)
                y = tf.squeeze(y, axis=1)

                # compute the loss (mse)
                loss = tf.square(y_onehot - out)
                # mean: scalar
                loss = tf.reduce_mean(loss)

            # compute gradients
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5])
            # w1 = w1 - lr * w1_grad
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])
            w4.assign_sub(lr * grads[6])
            b4.assign_sub(lr * grads[7])
            w5.assign_sub(lr * grads[8])
            b5.assign_sub(lr * grads[9])

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))


if __name__ == '__main__':
    train_db, test_db = mnist_dataset()
    my_nn(train_db)
