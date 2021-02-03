import tensorflow as tf

"""
where
scatter_nd
meshgrid
"""

"""
Where(tensor) tensor = True
"""

a = tf.random.normal([3, 3])

# [[True, True, False], [False, True, False], [True, False, True]]
mask = a > 0
a1 = tf.boolean_mask(a, mask=mask)  # 1D l=5 array

indices = tf.where(mask)
a2 = tf.gather_nd(a, indices=indices)

"""
where(cond, A, B)
       T/F matrix
choose A[i] or B[i] according to T/F matrix
"""


"""
scatter_nd
    indices,
    updates,
    shape    
background is a zeros array
"""

b = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])  # update 9, to index 4; update 10 to index 3...
shape = tf.constant([8])  # background is a l=8 all zeros array

c = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

"""
meshgrid

"""