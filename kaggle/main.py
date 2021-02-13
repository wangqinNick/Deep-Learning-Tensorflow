import numpy as np  # linear algebra
import pandas as pd  # data processing
import tensorflow as tf
import os
import datetime
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data():
    df = pd.read_csv('input/train.csv', encoding='big5')
    df_test = pd.read_csv('input/test.csv', encoding='big5')

    df.replace("NR", 0, inplace=True)  # replace all NR
    df_test.replace("NR", 0, inplace=True)

    df = df.drop('測站', axis=1)
    df = df.drop('日期', axis=1)

    df_test = df_test.drop('id_0', axis=1)
    df_test = df_test.drop('AMB_TEMP', axis=1)

    df = df.set_index('測項')
    df1_x = df.iloc[:, :9]
    df1_ = df.iloc[:, 9]

    df2_x = df.iloc[:, 9:18]
    df2_ = df.iloc[:, 18]

    df1_y = df1_.loc['PM2.5']
    df2_y = df2_.loc['PM2.5']

    data1_x = df1_x.to_numpy()
    data2_x = df2_x.to_numpy()
    data1_y = df1_y.to_numpy()
    data2_y = df2_y.to_numpy()

    df_test_x = df_test.to_numpy()

    data1_x.resize((240, 18 * 9))
    data2_x.resize((240, 18 * 9))

    df_test.loc['id', :] = df.iloc[0]
    df_test_x = df_test.to_numpy().astype(float)
    df_test_x.resize((240, 18 * 9))

    data_x = np.vstack((data1_x, data2_x)).astype(float)
    data_y = np.hstack((data1_y, data2_y)).astype(float)
    return (data_x, data_y), df_test_x


def preprocess(x_, y_):
    x_ = tf.cast(x_, dtype=tf.float32)
    y_ = tf.cast(y_, dtype=tf.float32)
    return x_, y_


if __name__ == '__main__':
    (x, y), x_test = load_data()
    print(x.shape, y.shape)

    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(10000).batch(100)

    y_test = np.zeros_like(x_test).astype(float)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(100)
    """
    db_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    db_test = db_test.map(preprocess).shuffle(10000).batch(5)
    """

    # [480, 162] -> [ 480, 1]
    model = Sequential([
        layers.Dense(80, activation=tf.nn.relu),
        layers.Dense(40, activation=tf.nn.relu),
        layers.Dense(20, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    model.build(input_shape=[None, 162])
    model.summary()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    optimizer = optimizers.Adam(lr=1e-3)
    loss_meter = metrics.Mean()

    for epoch in range(2000):
        # train
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss_mse = tf.reduce_mean(keras.losses.MSE(y_true=y, y_pred=logits))
                loss_meter.update_state(float(loss_mse))

            grads = tape.gradient(loss_mse, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

            with summary_writer.as_default():
                tf.summary.scalar('train-loss', loss_meter.result().numpy(), step=step)

                # evaluate
                # reset the loss_meter every 100 steps
                # loss_meter means to calculate the loss for the 100 steps
                loss_meter.reset_states()

        """# generate out
        total_correct = 0
        total_num = 0
        out = model(x)  # [BATCH_SIZE, 1]
        comp = out[:][0] - y
        # print(np.where(np.logical_and(comp > -5, comp < 5)))
        total_correct += len(np.where(np.logical_and(comp > -10, comp < 10))[0])
        total_num += y.shape[0]
        print(epoch, total_correct / total_num)"""
    y_pred = model(x_test)
    print(y_pred)
