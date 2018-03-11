import gym
import numpy as np
import random as rd
import math
import pydot
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Dropout, Flatten
from keras.utils import plot_model

Gam = 0.93
ep_min = 0.01
ep_decay = 1.0005
Iter = 1000
LRate = 0.001
d = False
batch_size = 40

class Agent:
    def __init__(self, acSize, obsSize):
        self.memory = deque(maxlen=3000)
        self.obsSize = obsSize
        self.acSize = acSize
        self.eps = 1.0

        #First Hidden layer (Input going to Hidden)
        self.Nnet = Sequential([Dense(12, input_dim=self.obsSize )])

        #Second Hidden layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(24, activation='relu'))

        #Third Hidden layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(48, activation='relu'))

        #Fourth Hidden layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(24, activation='relu'))

        #Fifth Hidden layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(12, activation='relu'))

        #Sixth Output Layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(self.acSize, activation='linear'))

        #mean square error
        self.Nnet.compile(loss='mse', optimizer=Adam(lr=LRate))


def train():
    for i in range(Iter):
        obs = env.reset()
        obs = np.reshape(obs, [1 , obsSize])
        for time in range(600):
            #env.render()
            if np.random.rand() > bot.eps:
                act = np.argmax(bot.Nnet.predict(obs)[0])
            else:
                act = rd.randrange(bot.acSize)
            next, rew, d, _ = env.step(act)
            if not d:
                rew = rew
            else:
                rew = -10
            next = np.reshape(next, [1, obsSize])
            bot.memory.append((act, rew, obs, next, d))
            obs = next
            if d:
                print("iteration: ",i,"/",Iter,"score:",time,"eps:",bot.eps)
                break
        minB = rd.sample(bot.memory, min(len(bot.memory), batch_size))
        for act, rew, st, next, d in minB:
            if d:
                tar = rew
            else:
                tar = (rew + Gam * np.amax(bot.Nnet.predict(next)[0]))

            fut = bot.Nnet.predict(st)
            fut[0][act] = tar
            bot.Nnet.fit(st, fut, epochs=1, verbose=0)
            if (ep_min - bot.eps) < 0:
                bot.eps /= ep_decay

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    obsSize = env.observation_space.shape[0]
    acSize = env.action_space.n
    print('***************************************')
    print('')
    print('Game Environment Information')
    print('\tInput/Environment size:', obsSize)
    print('\tOutput/Action size:', acSize)
    print('')
    print('***************************************')
    bot = Agent(acSize, obsSize)
    #plot_model(bot, to_file='model.png', show_shapes=True, show_layer_names=True)
    train()
