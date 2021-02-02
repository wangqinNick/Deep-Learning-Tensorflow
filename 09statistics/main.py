import tensorflow as tf

"""
tf.norm
tf.reduce_min/max
tf.argmax/argmin
tf.equal
tf.unique
"""

"""
vector norm
Eukl.Norm = sqrt(sum(xi^2))
L1-Norm = sum(abs(xi))
Max.norm = abs(max(xi))
"""
a = tf.random.uniform([2, 2])

a1 = tf.norm(a)

b = tf.ones([2, 2])  # [[1, 1], [1, 1]]
b1 = tf.norm(b, ord=2)
b2 = tf.norm(b, ord=2, axis=1)  # [sqrt2, sqrt2]
b3 = tf.norm(b, ord=1)

"""
reduce_min/max/mean
reduce dim: [[1, 2], [3, 4]] -> [1, 3]
"""

c = tf.random.normal([4, 10])

c_min = tf.reduce_min(c)  # min on all dims
c_min1 = tf.reduce_min(c, axis=1)  # min on dim 1 (10)
c_max = tf.reduce_max(c)
c_mean = tf.reduce_mean(c)

"""
tf.argmax: index of max (on axis n)
tf.argmin: index of min (on axis n)
"""

d = tf.random.normal([4, 10])
d1 = tf.argmax(d)
d2 = tf.argmax(d, axis=1)  # on last dim
d3 = tf.argmin(d)

"""
tf.equal
tf.reduce_sum
accuracy
"""

e = tf.constant([1, 2, 3, 4, 5])
f = tf.constant([1, 3, 4, 4, 6])

ef = tf.equal(e, f)  # [True, False, False, True, False]
eff = tf.reduce_sum(tf.cast(ef, tf.int32))  # cast True to 1; False to 0

"""
tf.unique
get all element (once, no duplicated)
Unique: non-dup elements
idx: idx of Unique
"""
g = tf.range(5)
g1 = tf.unique(g)  # Unique: [0, 1, 2, 3, 4] idx:[0, 1, 2, 3, 4]

h = tf.constant([4, 2, 2, 4, 3])
h1 = tf.unique(h)  # Unique: [4, 2, 3] idx:[0, 1, 1, 0, 2]
# h_ = tf.gather(h1)
