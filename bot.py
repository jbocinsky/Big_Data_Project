import gym
import numpy as np
import random as rd
import math
import pydot
import sys
import matplotlib.pyplot as plt

from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Dropout, Flatten
from keras.utils import plot_model

Gam = 0.93
ep_min = 0.01
ep_decay = 1.0003
Iter = 1000
LRate = 0.001
d = False
batch_size = 20

class Agent:
    def __init__(self, acSize, obsSize):
        self.memory = deque(maxlen=3000)
        self.obsSize = obsSize
        self.acSize = acSize
        self.eps = 1.0

        #First Hidden layer (Input going to Hidden)
        self.Nnet = Sequential([Dense(24, input_dim=self.obsSize )])

        #Second Hidden layer
        #self.Nnet.add(Dropout(0.01))
        #self.Nnet.add(Dense(36, activation='relu'))

        #Third Hidden layer
        #self.Nnet.add(Dropout(0.01))
        #self.Nnet.add(Dense(24, activation='relu'))

        #Fourth Hidden layer
        #self.Nnet.add(Dropout(0.01))
        # self.Nnet.add(Dense(18, activation='relu'))

        #Fifth Hidden layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(24, activation='relu'))

        #Sixth Output Layer
        #self.Nnet.add(Dropout(0.01))
        self.Nnet.add(Dense(self.acSize, activation='linear'))

        #mean square error
        self.Nnet.compile(loss='mse', optimizer=Adam(lr=LRate))


    # def loadNN(self, name):
    #         self.Nnet.load_weights(name)

    # def saveNN(self, name):
    #         self.Nnet.save_weights(name)

def train(game):
    times = np.arange(Iter)
    scores = np.zeros(Iter)
    for i in range(Iter):
        obs = env.reset()
        obs = np.reshape(obs, [1 , obsSize])
        
        for time in range(600):
            # env.render()

            if np.random.rand() > bot.eps:
                act = np.argmax(bot.Nnet.predict(obs)[0])
            else:
                act = rd.randrange(bot.acSize)

            if(game == 'Pendulum-v0'):
                next, rew, d, _ = env.step([act])
            else:
                next, rew, d, _ = env.step(act)
            if not d:
                rew = rew
            else:
                rew = -10

            next = np.reshape(next, [1, obsSize])
            bot.memory.append((act, rew, obs, next, d))
            obs = next

            if d:
                print("iteration:", i, "/", Iter, "   Score:", time, "   Eps:", bot.eps)
                sys.stdout.flush()
                scores[i] = time
                times[i] = i
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

    return scores, times
    #save final NN
    # bot.saveNN('./save/nnWeights.h5')

    

if __name__ == "__main__":
    game = 'CartPole-v1'
    # game = 'Pendulum-v0'
    # game = 'MountainCar-v0'
    # game = 'Acrobot-v1'
    env = gym.make(game)
    obsSize = env.observation_space.shape[0]
    if(game != 'Pendulum-v0'):
        acSize = env.action_space.n
    else:
        acSize = env.action_space.shape[0]

    print('***************************************')
    print('')
    print('Game Environment Information')
    print('\tInput/Environment size:', obsSize)
    print('\tOutput/Action size:', acSize)
    print('')
    print('***************************************')
    
    #create architecture
    bot = Agent(acSize, obsSize)
    #plot_model(bot, to_file='model.png', show_shapes=True, show_layer_names=True)

    #train architecture
    scores, times = train(game)

    #Print results
    plt.plot(times, scores)
    plt.title('Training Scores')
    plt.xlabel('iterations')
    plt.ylabel('scores')
    plt.show()
    # bot.loadNN('./save/nnWeights.h5')



