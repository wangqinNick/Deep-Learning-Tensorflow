import tensorflow as tf

# Indexing

""""
1. basic indexing: [idx0][idx1][idx2]...
2. same as numpy: [idx0, idx1, idx2]
"""
a = tf.random.normal([4, 28, 28, 3])  # 4 pictures
aa = a[1].shape
aaa = a[1, 2]
aaaa = a[1, 2, 3]

# Slicing
"""
returns a vector
start: end  [start: end)
single :
0, 1, 2
-1, -2, -3
"""
b = tf.range(10)
bb = b[-1:]  # last element
bbb = b[-2:]  # last two elements
bbbb = b[:-1]  # all but last elements
bbbbb = b[:2]  # first two elements

c = tf.random.normal([4, 28, 28, 3])  # 4 pictures
cc = c[0, :, :, :]  # all elements
c_r = c[:, :, :, 0]  # [4, 28, 28] every r of 4 pictures
c_b = c[:, :, :, 2]  # [4, 28, 28] every b
ccc = c[:, 0, :, :]  # [4, 28, 3]

"""
Indexing by step
double :
:step
"""

d = tf.random.normal([4, 28, 28, 3])
dd = d[0:2, :, :, :]  # every two picture
ddd = d[:, 0:28:2, 0:28:2, :]  # every two columns and rows
dddd = d[:, :14, :14, :]
dddd_v2 = d[:, 14:, 14:, :]
ddddd = d[:, ::2, ::2, :]
dddddd = d[:, 2:26:2, 2:26:2, :]

"""
reverse indexing
step < 0
::-1
[start, end)
"""

"""
...
represents any length of :
should not be ambiguous
"""
e = tf.random.normal([4, 28, 28, 3])
ee = e[0, ...]
ee_v2 = e[0, :, :, :]

ee_red = e[..., 0]
ee_green = e[..., 1]
eee = e[0, ..., 2]  # blue of the first picture
eeee = e[0, ..., 0]  # red of the first picture

# Selective indexing
"""
Gather from indexes
tf.gather(source, axis=which dimension, indices= index(s) on the dimension)

Gather from multiple dimensions
class1-student1, class2-student2, class3-student3
tf.gather_nd: [[0]]
tf.boolean_mask
"""

f = tf.random.normal([4, 35, 8])
f1 = tf.gather(f, axis=0, indices=[2, 3])
f1_v2 = f[2:4]

f2 = tf.gather(f, axis=0, indices=[2, 1, 3, 0])
f3 = tf.gather(f, axis=1, indices=[1, 3, 7])

g = tf.random.normal([4, 35, 8])
""" 
[[0], [], []]  -> [1, 35, 8]
[35, 8] 
"""
g1 = tf.gather_nd(g, [0])  # [35, 8]
g2 = tf.gather_nd(g, [0, 1])  # [8]
g3 = tf.gather_nd(g, [0, 1, 2])  # []
"""
[0, 1, 2] is a scalar
[[0,1,2]] is the vector of the scalar (wrapped in the vector)
"""
g4 = tf.gather_nd(g, [[0, 1, 2]])  # vector [1]

g5 = tf.gather_nd(g, [[0, 0], [1, 1]])  # [2, 8] ([[8] -> [8]])
g6 = tf.gather_nd(g, [[0, 0], [1, 1], [2, 2]])  # [3, 8]
g7 = tf.gather_nd(g, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])  # [3]
g7_in_vector = tf.gather_nd(g, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]])  # [1,3]


"""
boolean_mask
default on axis 0

mask:
True: select on axis-n
False: not select on axis-n

mask needs to be correspondent to the axis-n dimension num
"""
h = tf.random.normal([4, 28, 28, 3])
h1 = tf.boolean_mask(a, mask=[True, True, False, False])
h2 = tf.boolean_mask(a, mask=[True, True, False], axis=3)

i = tf.ones([2, 3, 4])

"""
first 2 dims forms a 2D array
[00]  01   02
 10  [11] [12]
 
 result -> [3, 4]
"""
i1 = tf.boolean_mask(i, mask=[[True, False, False], [False, True, True]])
