import numpy as np

import gym
from gym import spaces, logger
from gym.spaces import Discrete, Box
from gym.utils import seeding

from gym_ncs.envs.costfunctions import costfunction
from gym_ncs.envs.helpers import find_all_allocation_options


class SchedulingEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(p)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        p - 1        Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(n choose k)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
    """
    def __init__(self, system_parameters, period):
        # setting the period
        self.period = period
        # determining the action set
        self.action_set = find_all_allocation_options(system_parameters['no_plants'], system_parameters['no_channels'])
        # instantiating the cost function
        self.costfunc = costfunction(system_parameters)
        # defining action and observation spaces
        self.action_space = Discrete(len(self.action_set))
        self.observation_space = Box(low=-1, high=len(self.action_set), shape=(self.period,), dtype=np.int64)
        # initializing the seed
        self.seed()
        # initializing the state
        self.state = np.negative(np.ones((self.period,), dtype=np.int64))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if np.all(self.state >= 0):
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        for idx, element in enumerate(self.state):
            if element < 0:
                self.state[idx] = action
                break
        done = True if np.all(self.state >= 0) else False
        reward = self.normalize(self.costfunc(self.state)) if done else 0.0 # normalized reward
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.negative(np.ones((self.period,), dtype=np.int64))
        return self.state

    @staticmethod
    def normalize(value: float, max_value: float = 25000.0) -> float:
        return 0.0 if value > max_value else 1 - value / max_value
