# Periodic Scheduling: Independent closed-loop control systems

## Description

Consider a networked control system that consists of N independent feedback loops closed over a shared communication network that comprises M communication channels. Since the number of communication channels is strictly less than the number of subsystems, only a subset of feedback loops can be closed at each sampling interval. Therefore, a centralized scheduler orchestrates communication among entities (i.e., sensors and controllers) of these feedback control loops.

Each feedback control loop consists of an intelligent sensor, a controller, and an actuator. Each controller is collocated with an actuator but not with a sensor. Each sensor periodically takes noisy measurements of the subsystem's output at a fixed sampling rate. Then, each sensor computes the state estimates based on its measurements and transmits them to an associated remote controller whenever the scheduler allocates an available channel to this sensor. Each remote controller computes the control commands based on its estimates or the sensor's estimates (depending on the scheduler's decision) and sends the commands immediately to the actuator. Each actuator acts whenever it receives control commands. All data transmissions that take place in the networked control system are immediate and lossless.

The goal is to find a periodic communication sequence with an arbitrarily long period to optimize the overall control performance of the networked control system in terms of a quadratic function.

## Installation

### Building from source

```python
git clone https://github.com/demirelbu/gym-ncs.git
cd gym-ncs
pip install -e .
```

### Building from github

```python
pip install -e git+https://github.com/demirelbu/gym-ncs.git@master#egg=gym_ncs
```

## Example

Next, we provide an example to demonstrate how one can use the
proposed environment.

```python
from gym_ncs.envs.scheduling_env import SchedulingEnv
from gym_ncs.envs.control_systems import generate_random_systems


# Number of control systems
nUsers = 4
# Number of available channels
nChannels = 2
# Period of the communication sequence
period = 10
# Create random closed-loop systems
system_variables = generate_random_systems(dimensions=(
    2, 1, 1), nUsers=nUsers, stable=[True, False, False, False], seed=20)
# Instantiate the environment
env = SchedulingEnv(nUsers, nChannels, system_variables, period)
# Reset the environment
observation, done = env.reset(), False
print("Initial observation: {}, Done: {}".format(observation, done))
# Iterate until finding a complete communication sequence
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Observation: {}, Action: {}, Reward: {}, Done: {}".format(
        observation, action, reward, done))

```
