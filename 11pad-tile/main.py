import tensorflow as tf

"""
pad
tile
broadcast_to
"""

"""
pad: [left_len, right_len]
     [[up, down], [lef, rig]]
     [[0, 1], [1, 1]]
"""

a = tf.reshape(tf.range(9), [3, 3])

a1 = tf.pad(a, [[0, 0], [0, 0]])
a2 = tf.pad(a, [[1, 1], [1, 1]])
a3 = tf.pad(a, [[1, 0], [2, 1]])
a4 = tf.pad(a, [22, 22], [22, 22], constant_values=1)  # padding with const 1

"""
image padding
"""


"""
tile
repeat the data along dim n times
[a, b, c], 2 -> [a, b, c, a, b, c]
[dim1_n, dim2_n] 1 for not copy, 2 for twice...
inner-dim first
"""

b = tf.random.uniform([3, 3])

b1 = tf.tile(b, [1, 2])
b2 = tf.tile(b, [2, 1])
b3 = tf.tile(b, [2, 2])
