import numpy as np
from typing import Tuple, Dict, List
import numpy.typing as npt
from itertools import combinations


def find_all_allocation_options(nUsers: int, nChannels: int) -> Dict[int, Tuple[int, ...]]:
    """
    Description:
        This function finds all possible user allocations for the different number of user and channel combinations.

    Args:
        nUsers: An integer indicating the number of users.
        nChannels: An integer indicating the number of channels.

    Returns:
        dictionary: All possible combinations of plant and channel allocations.
    """
    all_user_allocations = list(combinations(
        [index for index in range(1, nUsers + 1)], nChannels))
    return {index: users for index, users in enumerate(all_user_allocations)}


def linear_search(indices: np.ndarray, user: int, nUsers: int, nChannels: int) -> Tuple[np.ndarray, bool]:
    """
    Description:
        This function performs a linear search on a list of integers to find a given integer.

    Args:
        indices: An index list (in np.ndarray with np.int64) whose elements take values from {0, 1, 2, ..., (nUsers choose nChannels)-1}.
        user: An integer indicating the user, i.e., taking values from {1, 2, 3, ..., nUsers}.
        nUsers: An integer indicating the number of users.
        nChannels: An integer indicating the number of channels.

    Returns:
    """
    all_user_allocations = find_all_allocation_options(nUsers, nChannels)
    _indices: np.ndarray = np.tile(indices, 2)
    _time: np.ndarray = np.zeros((2 * indices.shape[0],), dtype=np.int64)
    k, flag = 0, False  # user cannot be found
    for index, element in np.ndenumerate(_indices):
        if any([True if idx == user else False for idx in all_user_allocations[element]]):
            k, flag = 0, True  # user is found
        else:
            k += 1
        _time[np.squeeze(index)] = k
    return _time[indices.shape[0]:2 * indices.shape[0]], flag


def index_to_schedule(indices: List[int], nUsers: int, nChannels: int) -> List[Tuple[int, ...]]:
    """
    Description:
        This function converts a list of indices to a list of scheduling actions.

    Args:
        indices: An index list whose elements take values from {0, 1, 2, ..., [nUsers choose nChannels]-1}.
        nUsers: An integer indicating the number of users.
        nChannels: An integer indicating the number of channels.

    Returns:
    """
    all_user_allocations = find_all_allocation_options(nUsers, nChannels)
    return [all_user_allocations[index] for index in indices]
