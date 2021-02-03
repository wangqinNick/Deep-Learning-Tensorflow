import numpy as np

k = int(input())
biscuits = []
for i in input().split():
    q = int(i)
    biscuits.append(q)


def f(arr, n):
    if np.count_nonzero(arr) < n:
        return 0
    else:
        # eat the most one
        n += 1
        arr = arr.sort(reverse=True)
        for j in range(n):
            arr[j] = arr[j] - 1
        return 1 + f(arr, n)


s = f(arr=biscuits, n=1)
print(s)
