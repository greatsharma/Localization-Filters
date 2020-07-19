from __future__ import division
from __future__ import absolute_import

import random
import numpy as np

# check for valid probability distribution
def validate_dist(prob):
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            assert 0.0 <= prob[i][j] <=1.0, 'probabilities should be in range [0.0, 1.0]'

    assert 1.0 - np.sum(prob) <= (1e-8), 'probabilities should sum to 1.0'


def apply_padding(matrix, pad):
    (n_row, n_col) = matrix.shape
    matrix = np.vstack((matrix[n_row-pad:], matrix, matrix[0:pad]))
    matrix = np.hstack((matrix[:, n_col-pad:].reshape(n_row + 2*pad, pad), matrix,
                        matrix[:, 0:pad].reshape(n_row + 2*pad, pad)))

    (n_row, n_col) = matrix.shape
    matrix[0:pad, 0:pad] = matrix[n_row - 2*pad : n_row - pad, n_col - 2*pad : n_col - pad]
    matrix[0:pad, n_col - pad:] = matrix[n_row - 2*pad : n_row - pad, pad : 2*pad]
    matrix[n_row-pad:, 0:pad] = matrix[pad : 2*pad, n_col - 2*pad : n_col - pad]
    matrix[n_row-pad:, n_col-pad:] = matrix[pad : 2*pad, pad : 2*pad]

    return matrix


# convolution is just a weighted sum.
def convolution(matrix, kernel, padding=True):
    assert kernel.shape[0] == kernel.shape[1], 'kernel should be square'
    assert kernel.shape[0]%2 != 0, 'kernel shape should be odd'

    conv_matrix = np.zeros(matrix.shape)
    pad = kernel.shape[0] // 2

    if padding:
        matrix = apply_padding(matrix, pad)

    for i in range(conv_matrix.shape[0]):
        for j in range(conv_matrix.shape[1]):
            conv_matrix[i][j] = np.sum(np.multiply(matrix[i:(i + 2*pad + 1), j:(j + 2*pad + 1)], kernel))

    return conv_matrix
