from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from utils import validate_dist, convolution


# sense gains information hence decreases entropy, sense is just a product.
# sense calculates the change in prior due to the measurement using bayesian inference.
def sense(world, prior, measurement, pSense):
    assert world.shape == prior.shape, "world and prior should have same shape"
    validate_dist(prior)

    assert (0.0 <= pSense <= 1.0),
    "measurement sense probability should be in range [0.0, 1.0]"

    posterior = np.zeros_like(prior)
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            if world[i][j] == measurement:
                posterior[i][j] = prior[i][j] * pSense
            else:
                posterior[i][j] = prior[i][j] * (1.0 - pSense)

    normalizer = np.sum(posterior)
    posterior = np.asarray(list(map(lambda x: x / normalizer, posterior)))

    return posterior


# move losses information hence increases entropy.
# move is just weighted sum or convolution and is goverened by theorem of total probability.
def move(prior, motion, motionKernel):
    validate_dist(prior)
    validate_dist(motionKernel)

    posterior = np.zeros_like(prior)

    if np.array_equal(motion, [0, 0]):
        posterior = prior
    elif np.array_equal(motion, [0, 1]):
        posterior = convolution(prior, np.fliplr(motionKernel))
    elif np.array_equal(motion, [0, -1]):
        posterior = convolution(prior, motionKernel)
    elif np.array_equal(motion, [1, 0]):
        posterior = convolution(prior, np.rot90(motionKernel, k=3))
    elif np.array_equal(motion, [-1, 0]):
        posterior = convolution(prior, np.rot90(motionKernel, k=1))
    else:
        raise Exception("Invalid motion command {}".format(motion))

    return posterior


# localization is a repeated cycle of sense and move.
def apply_histrogram_filter(world, prior, measurements, motions, pSense, motionKernel):
    assert len(measurements) == len(motions),
    "number of measurements and motions should be same"

    prob = prior.copy()
    for measure, mov in zip(measurements, motions):
        prob = move(prob, mov, motionKernel)
        prob = sense(world, prob, measure, pSense)

    return prob


if __name__ == "__main__":
    world = np.full((4, 5), "r")
    world[0][1], world[0, 2], world[1][2], world[2][2], world[2][3] = ("g") * 5
    print("\nworld: \n", world)

    prior = np.full(world.shape, 1.0 / (world.shape[0] * world.shape[1]))
    print("\nprior: \n", prior)

    measurements = ["g"] * 5
    print("\nmeasurements: ", measurements)

    motions = np.array([[0, 0], [0, 1], [-1, 0], [-1, 0], [0, 1]])
    print("\nmotions: ", motions)

    pSense = 0.7
    motionKernel = np.array(
        [
            [0.0] * 5,
            [0.0, 0.005, 0.05, 0.10, 0.005],
            [0.0, 0.01, 0.03, 0.55, 0.09],
            [0.0, 0.005, 0.05, 0.10, 0.005],
            [0.0] * 5,
        ]
    )

    print("\npSense: ", pSense)
    print("\nmotion kernel: \n", motionKernel)

    posterior = apply_histrogram_filter(
        world, prior, measurements, motions, pSense, motionKernel
    )
    print("\nposterior: \n", posterior)
