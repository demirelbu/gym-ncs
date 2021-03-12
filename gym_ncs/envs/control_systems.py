import numpy as np
import scipy.linalg as LA
from typing import Any, Dict, List, Optional, Tuple


def generate_random_systems(dimensions: Tuple[int, int, int], nUsers: int, stable: List[bool], seed: Optional[int] = None) -> Dict[str, Any]:
    """Randomly generates system matrices, performance indices (of quadratic cost function),
    and covariance matrices of process and noise for given dimensions and number of systems.

    Args:
        dimensions: A tuple (of integers) comprising the dimension of system matrices, e.g., (n, m, p).
        nUsers: An integer indicating the number of control systems.
        stable: A list of logical variable that indicates whether or not the corresponding
          system is stable. The length of the list is equal to nUsers.
        seed: Optional; Random seed.

    Returns:
        A dictionary that contains all system matrices with desired properties.
    """
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    # Extracting the matrix dimensions
    n, m, p = dimensions
    # Create a python dictionary (Note: Here, instead of np.ndarray, Any is used as type)
    system_variables: Dict[str, Any] = {}

    for index in range(1, nUsers + 1):
        # System variables
        system_variables['A' +
                         str(index)] = generateSquareMatrix(n, stable[index-1])
        system_variables['B' + str(index)] = generateAnyMatrix(n, m)
        system_variables['C' + str(index)] = generateAnyMatrix(p, n)
        # Performance indices
        system_variables['Q' + str(index)] = generateSPDmatrix(n)
        system_variables['R' + str(index)] = generateSPDmatrix(m)
        # Covariance of process noise
        system_variables['W' + str(index)] = generateSPDmatrix(n)
        # Covariance of measurement noise
        system_variables['V' + str(index)] = generateSPDmatrix(p)
    return system_variables


def generateSquareMatrix(n: int, stable: bool = True, max_iter: int = 9999) -> np.ndarray:
    """This function generates a dense n x n matrix."""
    for ind in range(max_iter):
        M: np.ndarray = np.random.rand(n, n)
        eigvals = LA.eig(M)
        flag = True if np.max(np.abs(eigvals[0])) < 1 else False
        if flag == stable:
            return M
    print("Could not generate the desired matrix!")
    return np.zeros((n, n))


def generateAnyMatrix(n: int, m: int) -> np.ndarray:
    """This function generates a dense n x m matrix."""
    return np.random.rand(n, m)


def generateSPDmatrix(n: int) -> np.ndarray:
    """This function generates a dense n x n symmetric, positive definite matrix."""
    M = np.random.rand(n, n)
    return 0.5 * (M + M.transpose()) + n * np.eye(n)
