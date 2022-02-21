import gym
from tqdm import tqdm
from time import sleep
import numpy as np


class Frozen_Agent:
    """Agent Class for solving Frozen lake problem."""

    def __init__(self,alfa, gamma, epsilon):
        """Initialize method for Q learning algorithm."""
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make("FrozenLake-v1")
        # Stuff for defining Q Matrix
        print(self.env.observation_space.n)
        self.Q = np.random.uniform(0,-2, [self.env.observation_space.n, self.env.action_space.n])

    def play(self , epochs):
        print("training...")
        for i in tqdm(range(epochs)):
            done = False
            new_state = self.env.reset()
            while(not done):
                action = self.env.action_space.sample()
                new_state , reward, done , info = self.env.step(action)
                ant = self.Q[new_state , action]
                new_max = np.argmax(self.Q[new_state ,: ])
                new_value = (1-self.alfa) * ant  + self.alfa*(reward + self.gamma*new_max)
                self.Q[new_state , action] = new_value
                #self.env.render()
                #print(self.Q)
                #print(new_state)
                #sleep(1)


    def play_good(self):
        wins = 0
        for i in range(1000):
            done = False
            new_state = self.env.reset()
            while(not done):
                action = self.env.action_space.sample()
                new_state , reward, done , info = self.env.step(action)
                ant = self.Q[new_state , action]
                new_max = np.argmax(self.Q[new_state ,: ])
                new_value = (1-self.alfa) * ant  + self.alfa*(reward + self.gamma*new_max)
                self.Q[new_state , action] = new_value
                #self.env.render()
            if(reward == 1):
                #print("ganamos")
                wins += 1
            else:
                #print("perdimos")
                pass
        print(f"win prob = {wins/1000}")

if __name__ == "__main__":
    agent = Frozen_Agent(0.7, 0.70, 0.1)
    agent.play(10000)
    agent.play_good()


