import tensorflow as tf

"""
Broadcastable:
    advantages:
        Efficient
        Intuitive
    feature:     
        expand dim
        without copying data
    
[10] -> [b, 10]

tf.broadcast_to
tf.tile

key idea
Two steps:
    1) insert 1 dim ahead if needed
    2) expand dims with size 1 to same size

from right to left
[4, 16, 16, 32]
           [32]
-> 
step1
[1, 1, 1, 32]
-> 
step2
[4, 16, 16, 32]
"""

"""
Match from last dim
when has no axis:
[class, students, scores] + [scores] # scores: shape = [1]
when has a dim of 1:
[class, students, scores] + [students, 1]
"""

""""
Broadcastable:
Match from last (right) Dim:
    if current dim = 1, expand to the same as target
    if either has no dim, insert one dim and expand to the same
    otherwise, NOT Broadcastable

[4, 32, 14, 14]
[2, 32, 14, 14] not Broadcastable
[1, 32, 14, 14] Broadcastable
"""

x = tf.random.normal([4, 32, 32, 3])

y = tf.random.normal([3])
# y_ = tf.broadcast_to(y_, x_)
s = x + y  # broadcasting y_: [3] to [4, 32, 32, 3]

y1 = tf.random.normal([32, 32, 1])
# y1_ = tf.broadcast_to(y1, x_)
s1 = x + y1  # broadcasting y1: [32, 32, 1] to [4, 32, 32, 3]

y2 = tf.random.normal([4, 1, 1, 1])
# y2_ = tf.broadcast_to(y2, x_)
s2 = x + y2  # broadcasting y2: [4, 1, 1, 1] to [4, 32, 32, 3]

y3 = tf.random.normal([1, 4, 1, 1])
# s3 = x_ + y3  # cannot cast coz axis-1


"""
Broadcasting vs. Tile
"""