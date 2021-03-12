import numpy as np
import scipy.linalg as LA
from typing import Any, Dict
from gym_ncs.envs.helpers import linear_search


class CostFunction:
    """This code is for computing the infinite-horizon control cost (or loss) when periodically
        allocating multiple communication channels to multiple feedback control systems.
    """

    def __init__(self, nUsers: int, nChannels: int, variables: Dict[str, Any], maxvalue: float = 25000.0) -> None:
        """Inits CostFunction with the number of control systems, the number of channels,
        system matrices, and the maximum value of the cost function."""
        self.nUsers = nUsers
        self.nChannels = nChannels
        self.variables = variables
        self.maxvalue = maxvalue
        for ind_plant in range(1, self.nUsers + 1):
            # unpack system variables
            A = np.asmatrix(self.variables['A' + str(ind_plant)])
            B = np.asmatrix(self.variables['B' + str(ind_plant)])
            C = np.asmatrix(self.variables['C' + str(ind_plant)])
            Q = np.asmatrix(self.variables['Q' + str(ind_plant)])
            R = np.asmatrix(self.variables['R' + str(ind_plant)])
            W = np.asmatrix(self.variables['W' + str(ind_plant)])
            V = np.asmatrix(self.variables['V' + str(ind_plant)])
            # define a unit matrix with the dimension of A
            I = np.asmatrix(np.eye(A.shape[0]))
            # compute the algebraic Riccati equation (control problem)
            S = np.asmatrix(LA.solve_discrete_are(A, B, Q, R, e=None, s=None, balanced=True))
            # compute the control gains
            L = np.asmatrix(LA.inv(B.T * S * B + R) * B.T * S * A)
            # compute the cost index (related to the error)
            M = np.asmatrix(L.T * (B.T * S * B + R) * L)
            # update the controller's dictionary
            self.variables['S' + str(ind_plant)] = S
            self.variables['L' + str(ind_plant)] = L
            self.variables['M' + str(ind_plant)] = M
            # compute the algebraic Riccati equation (estimation problem)
            P = np.asmatrix(LA.solve_discrete_are(A.T, C.T, W, V, e=None, s=None, balanced=True))
            # compute the estimator (Kalman) gain
            K = np.asmatrix(P * C.T * LA.inv(C * P * C.T + V))
            # compute the estimator covariance
            F = (I - K * C) * P
            # update the estimator's dictionary
            self.variables['P' + str(ind_plant)] = P
            self.variables['K' + str(ind_plant)] = K
            self.variables['F' + str(ind_plant)] = F
            # compute the noise covariance
            N = K * C * P
            # update the noise's dictionary
            self.variables['N' + str(ind_plant)] = N

    def __call__(self, schedule: np.ndarray) -> float:
        cost: float = 0.0
        for plant_index in range(1, self.nUsers + 1):
            # unpack system variables
            A = np.asmatrix(self.variables['A' + str(plant_index)])
            N = np.asmatrix(self.variables['N' + str(plant_index)])
            S = np.asmatrix(self.variables['S' + str(plant_index)])
            W = np.asmatrix(self.variables['W' + str(plant_index)])
            F = np.asmatrix(self.variables['F' + str(plant_index)])
            M = np.asmatrix(self.variables['M' + str(plant_index)])
            # compute "elapsed time" sequence for a given plant
            time, flag = linear_search(schedule, plant_index, self.nUsers, self.nChannels)
            # find the length of the schedule
            period: float = len(time)
            if flag is False:
                eigvals = LA.eig(A)
                if np.max(np.abs(eigvals[0])) < 1:
                    X = np.asmatrix(LA.solve_discrete_lyapunov(A, N))
                    J = np.trace(S * W) + np.trace(F * M) + np.trace(X * M)
                else:
                    return self.maxvalue
            else:
                largest_time: int = max(time)
                _variables: Dict[str, Any] = {}
                _variables['Z0'] = np.asmatrix(np.zeros(N.shape))
                for time_index in range(1, largest_time + 1):
                    _variables['Z' + str(time_index)] = A * \
                        _variables['Z' + str(time_index - 1)] * A.T + N
                J = np.float64(0.0)
                for time_element in time:
                    J += np.trace(M * _variables['Z' + str(time_element)])
                J = np.trace(S * W) + np.trace(F * M) + J / period
            cost += float(J)
        return cost
