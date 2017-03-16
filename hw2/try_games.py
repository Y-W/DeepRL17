import gym

# env = gym.make('Enduro-v0')
env = gym.make('SpaceInvaders-v0')

for i_episode in range(20):
    observation = env.reset()
    t = 0
    while True:
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        t += 1
        # print t
        if done:
            print("Episode finished after {} timesteps".format(t))
            break
