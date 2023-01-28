from gym.envs.registration import register

register(id='Park_Car-v0', 
    entry_point='gym_park_car.envs:ParkCarEnv', 
)