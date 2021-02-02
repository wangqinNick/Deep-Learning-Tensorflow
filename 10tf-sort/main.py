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

res1 = tf.math.top_k(a, 2)
c_idx = res1.indices
c1 = res1.values

"""
top_k accuracy
prob: [0.1, 0.2, 0.3, 0.4]
correct label: [2]
only consider the top-1 prediction [3]       0%
only consider the top-2 prediction [3, 2]  100%
"""

prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])

target1 = tf.constant([2, 0])

k_b = tf.math.top_k(prob, 3).indices

# k_b: [2, 3] -> [3, 2]
k_b = tf.transpose(k_b, [1, 0])

# target: [2] -> [3, 2]
#         [2, 0] * 3
target1 = tf.broadcast_to(target1, [3, 2])


def accuracy(output, target, topk=(1,)):

    # max of k [1, 2, 3, ...]
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    # transpose
    pred = tf.transpose(pred, perm=[1, 0])

    # target * n
    target_ = tf.broadcast_to(target, pred.shape)

    # a matrix of T and F
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        # top k columns of correct
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)

    return res


output = tf.random.normal([10, 6])
# make sum of all eqs to 1
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)

print('prob: ', output.numpy())

pred = tf.argmax(output, axis=1)
print('pred: ', pred.numpy())
print('label ', target.numpy())

acc = accuracy(output, target, topk=(1, 2, 3, 4, 5, 6))
print('top_1-6 acc', acc)
