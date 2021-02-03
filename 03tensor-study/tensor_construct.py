import numpy as np
import tensorflow as tf

"""construct Tensor"""
# 1. from numpy, list
a = np.ones((2, 3))
aa = tf.convert_to_tensor(a)

b = [1, 2, 3]
bb = tf.convert_to_tensor(b)

c = [[1, 2, 3], [5, 6, 7]]
cc = tf.convert_to_tensor(c)

# 2. zeros, ones
"""[2, 3] is the shape of the tensor"""
dd = tf.zeros((2, 3), dtype=tf.float32)
ee = tf.zeros([2, 3], dtype=tf.int32)

ff = tf.ones([2, 2])
gg = tf.ones((3, 3), dtype=tf.float32)

"""construct a similar shape tensor"""
hh = tf.zeros_like(ff)
ii = tf.zeros(ff.shape)


"""construct a scalar 1"""
jj = tf.ones([])

# 3. fill
kk = tf.fill([2, 3], 1)
ll = tf.fill((2, 3), 1.0)

# 4. random
"""normal distribution"""
"""mean, stddev"""
mm = tf.random.normal([2, 2], mean=1, stddev=1)
"""truncated normal"""
nn = tf.random.truncated_normal([2, 2], mean=1, stddev=1)

"""uniform distribution"""
"""min, max"""
oo = tf.random.uniform([2, 2], minval=0, maxval=1)

"""random permutation"""
"""randomize two related series of data: index"""

# 5. constant
pp = tf.constant(1)
qq = tf.constant([2, 2])
rr = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)


# 6. Application
"""
[]: (scalar) loss, accuracy
[d]: (vector) bias(b)
[h, w]: (matrix) input x_, weight
[b, len, vec] vec: encoding length
[b, h, w, c] image:                      [num_pictures, height, width, 3(rgb)], feature maps
[t, b, h, w, c] meta-learning: [num_task, single_task...]
"""

