import tensorflow as tf
"""
Sort / argsort (on some dims)
Topk
Top-5 Acc.
"""

a = tf.random.shuffle(tf.range(5))

a1 = tf.sort(a, direction="DESCENDING")
a_idx = tf.argsort(a, direction="ASCENDING")
a_ = tf.gather(a, a_idx)

b = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)

b1 = tf.sort(b)  # default descending all dims separately
b_idx = tf.argsort(b)
"""
[[0, 1, 2],
 [1, 2, 0],
 [2, 0, 1]]
"""


"""
math.top_k
top k elements
"""
c = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)

res = tf.math.top_k(a, 2)
c_idx = res.indices
c1 = res.values

"""
top_k accuracy
prob: [0.1, 0.2, 0.3, 0.4]
correct label: [2]
only consider the top-1 prediction [3]       0%
only consider the top-2 prediction [3, 2]  100%
"""
