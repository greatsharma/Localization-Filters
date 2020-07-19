from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


def update_belief(mu_x, sig_x, mu_y, sig_y):
    sig_x += 1e-15  # tolerance
    sig_y += 1e-15

    mu_n = (np.multiply(sig_y, mu_x) +
            np.multiply(sig_x, mu_y)) / (sig_x + sig_y)
    sig_n = np.multiply(sig_x, sig_y) / (sig_x + sig_y)
    return (mu_n, sig_n)


def predict_position(mu_x, sig_x, mu_y, sig_y):
    mu_p = mu_x + mu_y
    sig_p = sig_x + sig_y
    return (mu_p, sig_p)


def kalman_position_estimation(measurements, motions, measure_sigma, motion_sigma, mu_i, sig_i):
    dim = len(mu_i)

    assert len(measurements) == len(motions), \
        'number of measurements and motions should be same'

    assert len(sig_i) == dim, \
        'intial mean and sigma should be of length number of dimensions'

    assert len(measure_sigma) == dim, \
        'measurement sigma should be of length number of dimensions'

    assert len(motion_sigma) == dim, \
        'motion sigma should be of length number of dimensions'

    mu = np.matrix(mu_i).T
    sig = np.matrix(sig_i).T
    measure_sigma = np.matrix(measure_sigma).T
    motion_sigma = np.matrix(motion_sigma).T

    for measure, mov in zip(measurements, motions):
        mu, sig = predict_position(mu, sig, mov.reshape(dim, 1), motion_sigma)
        mu, sig = update_belief(
            mu, sig, measure.reshape(dim, 1), measure_sigma)

    return (mu, sig)


def kalman_state_estimation(measurements, pos_i, u, P, F, H, R):
    dim = len(pos_i)

    assert P.shape[0] == P.shape[1] == 2 * dim, \
        'uncertainity matrix P should be of shape (2*dim, 2*dim)'

    assert F.shape[0] == F.shape[1] == 2 * dim, \
        'state transition matrix F should be of shape (2*dim, 2*dim)'

    assert R.shape[0] == R.shape[1] == dim, \
        'measurement noise matrix R should be of shape (dim, dim)'

    assert H.shape == (dim, 2*dim), \
        'measurement function matrix H should be of shape (1, 2*dim)'

    assert u.shape == (2*dim, 1), \
        'external motion vector u should be of shape (2*dim, 1)'

    x = np.zeros((2*dim, 1))
    for ind in range(x.shape[0]//2):
        x[ind, 0] = pos_i[ind]

    I = np.identity(P.shape[0])
    for measure in measurements:
        # prediction step
        x = F*x + u  # updating state, x' = x + v.dt + u
        P = F*P*F.T

        # measurement step
        z = np.matrix(measure)  # z = x1 + x2 + ...+ xn, n=dim
        e = z.T - H*x  # error
        S = H * P * H.T + R  # projecting system's uncertainity unto measurement space
        K = P * H.T * np.linalg.inv(S)  # kalman gain
        x = x + K*e
        P = (I - K*H)*P

    return (x, P)


if __name__ == '__main__':
    print("\n\nestimate 1D position: \n")
    measurements = np.array([[1.], [2.], [3.]])
    motions = np.array([[1], [1], [1]])
    measure_sigma = [1.]
    motion_sigma = [0.]
    mu_i = [0.]
    sig_i = [1000.]
    pos_mu, pos_sig = kalman_position_estimation(measurements, motions,
                                                 measure_sigma, motion_sigma, mu_i, sig_i)
    print(
        f"position estimate: {pos_mu}\nuncertanity in position estimate: {pos_sig}")

    print("\n\nestimate 2D position: \n")
    measurements = np.array([[5., 10.], [6., 8.], [7., 6.],
                             [8., 4.], [9., 2.], [10., 0.]])
    motions = np.array([[1, -2], [1, -2], [1, -2], [1, -2], [1, -2], [1, -2]])
    measure_sigma = [0.1, 0.1]
    motion_sigma = [0., 0.]
    mu_i = [4., 12.]
    sig_i = [0., 0.]
    pos_mu, pos_sig = kalman_position_estimation(measurements, motions,
                                                 measure_sigma, motion_sigma, mu_i, sig_i)
    print(
        f"position estimate: {pos_mu}\nuncertanity in position estimate: {pos_sig}")

    # motion state consists both position as well as speed
    print("\n\n1D motion state estimation: \n")
    measurements = np.array([[1], [2], [3]])
    dt = 1.
    pos_i = [0.]  # initial x position
    # external motion vector i.e., change for position and velocity due to external forces
    u = np.matrix([[0.], [0.]])
    P = np.matrix([[1000., 0.], [0., 1000.]])  # uncertainity covariance
    F = np.matrix([[1., dt], [0., 1.]])  # state transition matrix
    H = np.matrix([[1., 0.]])  # measurement function matrix
    R = np.matrix([[1.]])  # measurement noise
    state_mat, uncertain_mat = kalman_state_estimation(
        measurements, pos_i, u, P, F, H, R)
    print(
        f"state estimation: {state_mat}\nuncertanity in estimation: {uncertain_mat}")

    print("\n\n2D motion state estimation: \n")
    measurements = np.array([[5., 10.], [6., 8.], [7., 6.],
                             [8., 4.], [9., 2.], [10., 0.]])
    dt = 1.
    pos_i = [4., 12.]  # initial x, y positions
    # change for position and velocity due external forces
    u = np.zeros((2*len(pos_i), 1))
    P = np.matrix([[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 1000., 0.],
                   [0., 0., 0., 1000.]])
    F = np.matrix([[1., 0., dt, 0.],
                   [0., 1., 0., dt],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    H = np.matrix([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])
    R = np.matrix([[0.1, 0.],
                   [0., 0.1]])
    state_mat, uncertain_mat = kalman_state_estimation(
        measurements, pos_i, u, P, F, H, R)
    print(
        f"state estimation: {state_mat}\nuncertanity in estimation: {uncertain_mat}")
