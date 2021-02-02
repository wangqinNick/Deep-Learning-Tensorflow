import tensorflow as tf

"""
Tensor operations:
tf.concat +
tf.split -
tf.stack +
tf.unstack -
"""

"""
concat: [a, b], axis=0
[class 1-4, students, scores] [4, 35, 8]
[class 5-6, students, scores] [2, 35, 8]
on axis 0                   ->[6, 35, 8]
"""

a = tf.ones([4, 35, 8])
b = tf.random.truncated_normal([2, 35, 8])
ab = tf.concat([a, b], axis=0)

a1 = tf.zeros([4, 32, 8])
b1 = tf.random.uniform([4, 3, 8])
a1b1 = tf.concat([a1, b1], axis=1)

"""
stack: create new dim
[c, d], axis=0
school1:[classes, students, scores]    [4, 35, 8]
school2:[classes, students, scores]    [4, 35, 8]
->
[school, classes, students, scores] [2, 4, 35, 8]
"""

c = tf.random.truncated_normal([4, 35, 8])
d = tf.random.uniform([4, 35, 8])

cd = tf.stack([c, d], axis=0)  # [2, 4, 35, 8]
cd1 = tf.stack([c, d], axis=3)  # [4, 35, 8, 2]

"""DIM MISMATCH"""

"""
unstack
size = 1 (fixed)
"""
c1, d1 = tf.unstack(cd, axis=0)  # [4, 35, 8]
cd_list1 = tf.unstack(cd, axis=3)  # 8 x [2, 4, 35]

""""
split
(c, axis=0, num_or_size_splits=2)
"""

cd_list2 = tf.split(cd, axis=3, num_or_size_splits=2)
cd_list3 = tf.split(cd, axis=3, num_or_size_splits=[2, 2, 4])

