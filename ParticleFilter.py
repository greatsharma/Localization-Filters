from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import random
from math import *


class Robot():

    def __init__(self, map: list, landmarks: list = None):
        self.map = map
        self.x_range = map[0]
        self.y_range = map[1]
        self.x_pos = random.uniform(0., self.x_range)
        self.y_pos = random.uniform(0., self.y_range)
        self.orientation = random.uniform(0.0, 2.0*pi)
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

        if landmarks:
            if any(len(lm) != 2 for lm in landmarks):
                raise ValueError(
                    'all landmarks should of length 2 containing [x_pos, y_pos]')
            else:
                self.landmarks = landmarks

    def set_position(self, pos: list):
        if pos[0] < 0. or pos[0] > self.x_range:
            raise ValueError('x position out of range')
        if pos[1] < 0. or pos[1] > self.y_range:
            raise ValueError('y position out of range')
        if pos[2] < -2*pi or pos[2] > 2*pi:
            raise ValueError('orientation should be between [-2pi...2pi]')

        self.x_pos = pos[0]
        self.y_pos = pos[1]
        self.orientation = pos[2]

    def set_noise(self, noise: list):
        if any(n < 0. for n in noise):
            raise ValueError('noise cannot be negative')

        self.forward_noise = noise[0]
        self.turn_noise = noise[1]
        self.sense_noise = noise[2]

    def sense(self):
        sensor_z = []
        for lm in self.landmarks:
            dist = sqrt((self.x_pos-lm[0])**2 + (self.y_pos-lm[1])**2)
            dist += random.gauss(0.0, self.sense_noise)
            sensor_z.append(dist)

        return sensor_z

    def move(self, turn, forward):
        if forward < 0.:
            raise ValueError('robot cannot move backwards')
        if turn < -2*pi or turn > 2*pi:
            raise ValueError('orientation should be between [-2pi...2pi]')

        self.orientation += turn + random.gauss(0.0, self.turn_noise)
        self.orientation %= 2*pi
        dist_moved = forward + random.gauss(0., self.forward_noise)
        self.x_pos += cos(self.orientation) * dist_moved
        self.x_pos %= self.x_range
        self.y_pos += sin(self.orientation) * dist_moved
        self.y_pos %= self.y_range

    def measurement_likelihood(self, sensor_z: list):
        if len(sensor_z) != len(self.landmarks):
            raise ValueError('incomplete sensor measurements')

        likelihood = 1.
        for z, lm in zip(sensor_z, self.landmarks):
            dist = sqrt((self.x_pos-lm[0])**2 + (self.y_pos-lm[1])**2)
            likelihood *= exp(-((dist - z)**2)/(self.sense_noise**2) /
                              2.) / sqrt(2.0 * pi * (self.sense_noise ** 2))

        return likelihood

    def __getitem__(self, index):
        if index == 0:
            return self.x_pos
        elif index == 1:
            return self.y_pos
        elif index == 2:
            return self.orientation
        else:
            raise ValueError('invalid index')

    def __setitem__(self, index, val):
        if index == 0:
            if val < 0. or val > self.x_range:
                raise ValueError('x position out of range')
            self.x_pos = val
        elif index == 1:
            if val < 0. or val > self.y_range:
                raise ValueError('y position out of range')
            self.y_pos = val
        elif index == 2:
            if val < -2*pi or val > 2*pi:
                raise ValueError('orientation should be between [-2pi...2pi]')
            self.orientation = val
        else:
            raise ValueError('invalid index')

    def __eq__(self, robot):
        if not isinstance(robot, Robot):
            return False

        return self.x_pos == robot.x_pos and self.y_pos == robot.y_pos

    def __repr__(self):
        return '[x_pos: {}  y_pos: {}  orient: {}]'.format(self.x_pos, self.y_pos, self.orientation)


def eval(robot, particles, map):
    sum = 0.
    for p in particles:
        dx = ((p.x_pos - robot.x_pos + map[0]/2.) % map[0]) - map[0]/2.
        dy = ((p.y_pos - robot.y_pos + map[1]/2.) % map[1]) - map[1]/2.
        err = sqrt(dx*dx + dy*dy)
        sum += err

    return sum / len(particles)


def apply_particle_filter(map, landmarks, n_particles, n_iter):
    robot = Robot(map, landmarks)
    # robot.set_noise([0.05, 0.05, 5.])

    # initalize particles
    particles = []
    for i in range(n_particles):
        p = Robot(map, landmarks)
        p.set_noise([0.05, 0.05, 5.])
        particles.append(p)

    # run particle filter
    for iter in range(n_iter):
        robot.move(0.1, 5.)
        sensor_z = robot.sense()

        weights = []
        for p in particles:
            p.move(0.1, 5.)
            weights.append(p.measurement_likelihood(sensor_z))

        w_norm = sum(weights)
        weights = [w/w_norm for w in weights]

        # resampling
        particles = random.choices(particles, weights, k=n_particles)

    return robot, particles


if __name__ == '__main__':
    map = [100, 100]
    landmarks = [[20., 20.], [80., 80], [20., 80.], [80., 20.]]
    robot, particles = apply_particle_filter(map, landmarks, 1000, 10)

    print(eval(robot, particles, map))
