import tensorflow as tf
"""
clip_by_value
relu
clip_by_norm
gradient clipping
"""

a = tf.random.uniform([10, ])

# return max(ai, 2)
a1 = tf.maximum(a, 2)

# return min(a, 2)
a2 = tf.minimum(a, 2)

# a combination
# return min(ai, 2) and max(ai, 8)
a3 = tf.clip_by_value(a, 2, 8)


"""
relu(a): return max(ai, 0)
"""
b = tf.range(10)

b1 = tf.nn.relu(b)


"""
clip_by_norm
"""
c = tf.random.normal([2, 2], mean=10)
print(tf.norm(c))
c1 = tf.clip_by_norm(a, tf.norm(c)/2)
print(tf.norm(c1))


""""
gradient clipping

gradient exploding: too large step
gradient vanishing: too small step

all gradients, same scale
tf.clip_by_global_norm
"""
