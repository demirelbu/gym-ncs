import numpy as np
from typing import Tuple, Dict, List
from itertools import combinations


def find_all_allocation_options(nUsers: int, nChannels: int) -> Dict:
    """
    Description:
        This function finds all possible user allocations for the different number of user and channel combinations.
    """
    all_user_allocations = list(combinations([index for index in range(1, nUsers + 1)], nChannels))
    return {index: users for index, users in enumerate(all_user_allocations)}


def linearsearch(indices: np.ndarray, user: int, nUsers: int, nChannels: int) -> [np.ndarray, bool]:
    """
    Description:
        This function performs a linear search on a list of integers to find a given integer.
    """
    all_user_allocations = find_all_allocation_options(nUsers, nChannels)
    _indices = np.tile(indices, 2)
    _time = np.zeros((2 * indices.shape[0],), dtype=np.int64)
    k, flag = 0, False  # user cannot be found
    for index, element in enumerate(_indices):
        if any([True if idx == user else False for idx in all_user_allocations[element]]):
            k, flag = 0, True  # user is found
        else:
            k += 1
        _time[index] = k
    return _time[indices.shape[0]:2 * indices.shape[0]], flag


def index2schedule(indices: List[int], nUsers: int, nChannels: int) -> List[int]:
    """
    Description:
        This function converts a list of indices to a list of scheduling actions.
    """
    all_user_allocations = find_all_allocation_options(nUsers, nChannels)
    return [all_user_allocations[index] for index in indices]
