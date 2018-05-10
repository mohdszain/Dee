import gym
environment = gym.make('CartPole-v0')
for i in range(50):
    test = environment.reset()
    for t in range(100):
        environment.render()
        print(test)
        action = environment.action_space.sample()
        test, reward, done, info = environment.step(action)

#Referred from: https://gym.openai.com/docs/