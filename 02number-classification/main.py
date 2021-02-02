import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# xs is the 60k numbers image training data, and each image is a 2D array [29, 28]
# xs = [60K, 28, 28]
# ys is the actual value of the image: 0, 1, 2, ..., 9
# ys = [60k]
# x_val and y_val are the correspondent test data
(xs, ys), (x_val, y_val) = datasets.mnist.load_data()
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
ys = tf.convert_to_tensor(ys, dtype=tf.int32)

# convert the y value to one_hot value
ys = tf.one_hot(ys, depth=10)

# change the the Dataset object, so that the GPU can compute a batch of data at one time
# a number of images processed at the same time (parallel)
train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys))

# process 200 individuals [a batch] at the same time
train_dataset = train_dataset.batch(200)

# the neural network
# apply three linear regression for every batch
# resize the data from [1,768] to [1, 10]
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)  # the optimizer for each linear regression


def train_epoch(epoch):
    """
    train the entire dataset
    :param epoch: the entire dataset
    """
    # Step4.loop [total size / batch times = 60k / 200 = 300 times] for the whole dataset
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            # apply the defined neural network
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
