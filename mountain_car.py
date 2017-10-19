import gym

env = gym.make('MountainCar-v0')
env.reset()
print(env.action_space)

for i in range(2000):
    env.render()
    env.step(env.action_space.sample())
