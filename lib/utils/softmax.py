import numpy as np


def softmax(arr):
    softmax_arr = np.zeros_like(arr).astype(np.float32)
    N = len(arr)

    for i in range(N):
        V = arr[i]

        exp = np.exp(V)
        softmax_out = exp/np.sum(exp)
        softmax_arr[i] = softmax_out

    return softmax_arr