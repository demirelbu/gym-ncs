import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym_ncs.envs.cost_function import CostFunction
from gym_ncs.envs.helpers import find_all_allocation_options
from typing import Any, Dict, List, Optional, Tuple, Union


class SchedulingEnv(gym.Env):
    """
    Description:
        A networked control system consists of N independent linear feedback
        control loops, sharing a communication network with M channels (M<N).
        A scheduling protocol that produces periodic communication sequences,
        dictates which feedback loops should utilize all available channels.
        The goal is to find the periodic communication sequence that minimizes
        the overall control loss of the networked control system.

    Source:
        This environment corresponds to the version of the periodic scheduling problem
        described by Demirel and Aytekin.

    Observation:
        Type: Box(p)
        Num     Observation                         Min                    Max
        0       First element of com. sequence       0                     (n choose k)-1
        .
        .
        .
        p-1     Last element of com. sequence        0                     (n choose k)-1

        Note: Here, p denotes the period of the communication seqquence.

    Actions:
        Type: Discrete(n choose k)
        Num                 Action
        0                   First element of the action sequence
        .
        .
        .
        (n choose k)-1      Last element of the action sequence

        Note: Action sequence contains all possible combinations of control system and channel assignment.
        Here, n denotes the number of control systems while k denotess the number of channels.

    Reward:
        Reward is either a floating point number which is less than max_value or zero.

    Starting State:
        All elements in the periodic communication sequence are set to -1.

    Episode Termination:
        When the last element in the periodic communication sequence is chosen.
    """

    def __init__(self, nUsers: int, nChannels: int, system_parameters: Dict[str, Any], period: int) -> None:
        super(SchedulingEnv, self).__init__()
        # setting the period
        self.period = period
        # determining the action set
        self.action_set = find_all_allocation_options(nUsers, nChannels)
        # instantiating the cost function
        self.costfunc = CostFunction(nUsers, nChannels, system_parameters)
        # defining action and observation spaces
        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            low=-1, high=len(self.action_set), shape=(self.period,), dtype=np.int64)
        # initializing the seed
        self.seed()
        # initializing the state
        self.state: np.ndarray = np.negative(
            np.ones((self.period,), dtype=np.int64))

    def seed(self, seed: Optional[int] = None) -> List[Union[int, None]]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Any]:
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        if np.all(self.state >= 0):
            logger.warn("You are calling 'step()' even though this environment \
                has already returned done = True. You should always call 'reset()' \
                    once you receive 'done = True' -- any further steps are undefined behavior.")

        for idx, element in enumerate(self.state):
            if element < 0:
                self.state[idx] = action
                break
        done = True if np.all(self.state >= 0) else False
        reward = self.normalize(self.costfunc(
            self.state)) if done else 0.0  # normalized reward
        return self.state, reward, done, {}

    def reset(self) -> np.ndarray:
        self.state: np.ndarray = np.negative(np.ones((self.period,), dtype=np.int64))
        return self.state

    @staticmethod
    def normalize(value: float, max_value: float = 25000.0) -> float:
        return 0.0 if value > max_value else 1 - value / max_value
