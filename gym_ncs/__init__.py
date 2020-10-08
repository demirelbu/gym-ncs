from gym.envs.registration import register

register(
    id='PeriodicSchedule-v0',
    entry_point='gym_ncs.envs:ScheduleEnv',
)
