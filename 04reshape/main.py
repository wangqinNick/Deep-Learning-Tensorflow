import tensorflow as tf

"""
[b, h, w]
View:

view 1: [b, 28, 28]
view 2: [b, 28*28] treat 28*28 as one dimension
view 3: [b, 2, 14*28]
view 4: [b, 28, 28, 1]
"""

a = tf.random.normal([4, 28, 28, 3])
a1 = tf.reshape(a, [4, 784, 3])
a2 = tf.reshape(a, [4, -1, 3])  # auto compute
a3 = tf.reshape(a, [4, 784 * 3])
a4 = tf.reshape(a, [4, -1])


"""
Transpose:

[b, h, w, c]

Change the content
Switch h, w
"""

b = tf.random.normal([4, 28, 28, 3])
b1 = tf.transpose(a)
b2 = tf.transpose(a, perm=[0, 2, 1, 3])  # perm[previous x_ dim]


""""
Expand
new dim (shape = 1)
"""
c = tf.random.normal([4, 35, 8])
c1 = tf.expand_dims(a, axis=0)  # [1, 4, 35, 8]
c2 = tf.expand_dims(a, axis=3)   # [4, 35, 8, 1]
c3 = tf.expand_dims(a, axis=-1)  # [4, 35, 8, 1]
c4 = tf.expand_dims(a, axis=-4)  # [1, 4, 35, 8]

"""
Squeeze
delete dim (shape = 1)
"""
d = tf.random.normal([1, 4, 35, 8, 1])
d1 = tf.squeeze(d)  # [4, 35, 8]
d2 = tf.squeeze(d, axis=0)
d3 = tf.squeeze(d, axis=-1)
