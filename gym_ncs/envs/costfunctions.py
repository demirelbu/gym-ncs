import numpy as np
from scipy import linalg as LA
from typing import Tuple, Dict, List

from gym_ncs.envs.helpers import linearsearch

class costfunction:
    """
    Description:
        This code is for computing the infinite-horizon control cost (or loss) when periodically
        allocating multiple communication channels to multiple feedback control systems.
    """
    def __init__(self, variables: Dict, maxvalue: float = 25000.0) -> None:
        #
        self.no_plants: int = variables['no_plants']
        self.no_channels: int = variables['no_channels']
        self.variables: Dict = variables
        self.maxvalue: float = maxvalue
        for ind_plant in range(1, self.no_plants + 1):
            # unpack system variables
            A = self.variables['A' + str(ind_plant)]
            B = self.variables['B' + str(ind_plant)]
            C = self.variables['C' + str(ind_plant)]
            Q = self.variables['Q' + str(ind_plant)]
            R = self.variables['R' + str(ind_plant)]
            W = self.variables['W' + str(ind_plant)]
            V = self.variables['V' + str(ind_plant)]
            # define a unit matrix with the dimension of A
            I = np.eye(A.shape[0])
            # compute the algebraic Riccati equation (control problem)
            S = LA.solve_discrete_are(
                A, B, Q, R, e=None, s=None, balanced=True)
            # compute the control gains
            L = LA.inv(B.T * S * B + R) * B.T * S * A
            # compute the cost index (related to the error)
            M = L.T * (B.T * S * B + R) * L
            # update the controller's dictionary
            self.variables['S' + str(ind_plant)] = S
            self.variables['L' + str(ind_plant)] = L
            self.variables['M' + str(ind_plant)] = M
            # compute the algebraic Riccati equation (estimation problem)
            P = LA.solve_discrete_are(
                A.T, C.T, W, V, e=None, s=None, balanced=True)
            # compute the estimator (Kalman) gain
            K = P * C.T * LA.inv(C * P * C.T + V)
            # compute ...
            F = (I - K * C) * P
            # update the estimator's dictionary
            self.variables['P' + str(ind_plant)] = P
            self.variables['K' + str(ind_plant)] = K
            self.variables['F' + str(ind_plant)] = F
            # compute the noise covariance
            N = K * C * P
            # update the noise's dictionary
            self.variables['N' + str(ind_plant)] = N

    def __call__(self, schedule: List[int]) -> float:
        cost: float = 0.0
        for plant_index in range(1, self.no_plants + 1):
            # unpack system variables
            A = self.variables['A' + str(plant_index)]
            N = self.variables['N' + str(plant_index)]
            S = self.variables['S' + str(plant_index)]
            W = self.variables['W' + str(plant_index)]
            F = self.variables['F' + str(plant_index)]
            M = self.variables['M' + str(plant_index)]
            # compute "elapsed time" sequence for a given plant
            time, flag = linearsearch(schedule, plant_index, self.no_plants, self.no_channels)
            # find the length of the schedule
            period: float = len(time)
            if flag is False:
                eigenvalues, _ = LA.eig(A)
                if np.max(np.absolute(eigenvalues)) < 1:
                    X = LA.solve_discrete_lyapunov(A, N)
                    J = np.trace(S * W) + np.trace(F * M) + np.trace(X * M)
                else:
                    return self.maxvalue
            else:
                largest_time: int = max(time)
                _variables: Dict = {}
                _variables['Z0'] = np.zeros(N.shape)
                for time_index in range(1, largest_time + 1):
                    _variables['Z' + str(time_index)] = A * \
                        _variables['Z' + str(time_index - 1)] * A.T + N
                J: float = 0.0
                for time_element in time:
                    J += np.trace(M * _variables['Z' + str(time_element)])
                J = np.trace(S * W) + np.trace(F * M) + J / period
            cost += J
        return cost

    def __str__(self) -> str:
        return f"This function computes the cost."
