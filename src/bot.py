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

Gam = 0.99
ep_min = 0.01
ep_decay = 1.00015
episodes = 1000
maxGameFrames = 1000
LRate = 0.001
d = False
batchSize = 64

class Agent:
    def __init__(self, acSize, stateSize):
        self.memory = deque(maxlen=2000)
        self.stateSize = stateSize
        self.acSize = acSize
        self.eps = 1.0
        self.startTraining = 1000

        #Make online and target model
        self.onlineModel = self.constructModel()
        self.targetModel = self.constructModel()

        #Copy weights from online to target model
        self.updateTargetModel()
        

    def constructModel(self):
        #First Hidden layer (Input going to Hidden)
        model = Sequential([Dense(24, input_dim=self.stateSize)])
        model.add(Dropout(0.2))

        # Second Hidden layer
        model.add(Dense(36, activation='relu'))
        model.add(Dropout(0.2))


        # Third Hidden layer
        model.add(Dense(48, activation='relu'))
        model.add(Dropout(0.2))


        # Fourth Hidden layer
        model.add(Dense(36, activation='relu'))
        model.add(Dropout(0.2))


        #Fifth Hidden layer
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.2))

        #Sixth Output Layer
        model.add(Dense(self.acSize, activation='linear', kernel_initializer='he_uniform'))

        model.summary()
        #mean square error
        model.compile(loss='mse', optimizer=Adam(lr=LRate))
        return model

    def trainOnlineModel(self):
        #Ensure we've seen enough observations, before we start training
        if(self.startTraining > len(self.memory)):
            return

        #Number of observations we will get for our mini batch
        numObservations = min(len(self.memory), batchSize)
        miniBatch = rd.sample(self.memory, numObservations)

        #Variables that will be passed in to target model for evaluation
        inputState = np.zeros((numObservations, self.stateSize))
        outputTarget = np.zeros((numObservations, self.stateSize))
        action = []
        reward = []
        done = []

        #Order of memory (act, reward, state, nextState, done)
        #Grab specifics out of memory of miniBatch
        for obs in range(numObservations):
            action.append(miniBatch[obs][0])
            reward.append(miniBatch[obs][1])
            inputState[obs] = miniBatch[obs][2]
            outputTarget[obs] = miniBatch[obs][3]
            done.append(miniBatch[obs][4])

        #Predict values through online and target network
        predictedTarget = self.onlineModel.predict(inputState)
        actualTarget = self.targetModel.predict(outputTarget)


        for obs in range(numObservations):
            if done[obs]:
                predictedTarget[obs][action[obs]] = reward[obs]
            else:
                #Q-Learning use max value for target
                predictedTarget[obs][action[obs]] = (reward[obs] + Gam * np.amax(actualTarget[obs]))


        self.onlineModel.fit(inputState, predictedTarget, batch_size=numObservations, epochs=1, verbose=0)


    def updateTargetModel(self):
        self.targetModel.set_weights(self.onlineModel.get_weights())


    # def loadNN(self, name):
    #         self.Nnet.load_weights(name)

    # def saveNN(self, name):
    #         self.Nnet.save_weights(name)

def playFinalGame(game, bot):
    state = env.reset()
    state = np.reshape(state, [1 , stateSize])
    done = False
    score = 0

    while not done:
        env.render()

        #get action
        qValue = bot.onlineModel.predict(state)
        act = np.argmax(qValue[0])

        #Take the action    
        if(game == 'Pendulum-v0'):
            nextState, reward, done, _ = env.step([act])
        else:
            nextState, reward, done, _ = env.step(act)

        #Set reward        
        if(game == 'CartPole-v1'):
            reward = 1

        if(game == 'MountainCar-v0'):
            #Set reward to be the velocity of the cart to encourage it to move fast to reach mountain
            reward = np.abs(nextState[1])
            #If beat the game, acquire large reward
            if(nextState[0] >= .5):
                reward = 10


        nextState = np.reshape(nextState, [1, stateSize])
        state = nextState
        score += reward

    print('Final game score:', score)


def train(game, bot, renderTraining):
    #Initialize for plotting
    times = []
    scores = []
    
    for epi in range(episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1 , stateSize])
        
        while not done:
            if renderTraining:
                env.render()

            #Get an action from learned NN action
            if np.random.rand() > bot.eps:
                qValue = bot.onlineModel.predict(state)
                act = np.argmax(qValue[0])
            #Get a random action
            else:
                act = rd.randrange(bot.acSize)

            #Take the action    
            if(game == 'Pendulum-v0'):
                nextState, reward, done, _ = env.step([act])
            else:
                nextState, reward, done, _ = env.step(act)

            #Set reward
            if(game == 'MountainCar-v0'):
                #Set reward to be the velocity of the cart to encourage it to move fast to reach mountain
                reward = np.abs(nextState[1])
                #If beat the game, acquire large reward
                if(nextState[0] >= .5):
                    reward = 10

            if(game == 'CartPole-v1'):
                #Set reward   
                if not done:
                    reward = reward
                    # print("reward:", reward)
                    # sys.stdout.flush()

                else:
                    reward = -100


            nextState = np.reshape(nextState, [1, stateSize])

            #Save observations to memory
            bot.memory.append((act, reward, state, nextState, done))
            
            #Update epsilon by decay rate
            if (ep_min - bot.eps) < 0:
                bot.eps /= ep_decay

            #Use miniBatch from memory to train model
            bot.trainOnlineModel()

            score += reward
            state = nextState

            #If game over
            if done:
                #Update target model
                bot.updateTargetModel()

                #Adjust score for applying a -100 reward if the game failed
                if(game == 'CartPole-v1'):
                    if score == 500:
                        score = score
                    else:
                        score = score + 100


                #save results for plotting and print to console
                print("episode:", epi, "/", episodes, "   Score:", score, "   Eps:", bot.eps)
                sys.stdout.flush()
                scores.append(score)
                times.append(epi)

                if(game == 'CartPole-v1'):
                    #if succeeded 10 times in a row
                    if np.mean(scores[-min(10, len(scores)):]) > 495:
                        print("Completed training!")
                        return bot, scores, times

                if(game == 'MountainCar-v0'):
                    #reward for getting to the top of the hill is 10, make sure last 15 made it to top
                    if all(i >= 10 for i in scores[-20:]):
                        print("Completed training!")
                        return bot, scores, times

                    # if np.mean(scores[-min(15, len(scores)):]) > 13:
                    #     print("Completed training!")
                    #     return bot, scores, times

    print("All:", episodes, "completed")
    print("Did not find optimal solution")
    return bot, scores, times

    
    #save final NN
    # bot.saveNN('./save/nnWeights.h5')

    

if __name__ == "__main__":

    #Settings:
    # game = 'CartPole-v1'
    # game = 'Pendulum-v0'
    game = 'MountainCar-v0'
    # game = 'Acrobot-v1'

    renderTraining = False

    env = gym.make(game)
    env._max_episodes = maxGameFrames
    stateSize = env.observation_space.shape[0]
    if(game != 'Pendulum-v0'):
        acSize = env.action_space.n
    else:
        acSize = env.action_space.shape[0]

    print('***************************************')
    print('')
    print('Playing', game)
    print('Wish me luck :)')
    print('')
    print('Game Environment Information')
    print('\tInput/Environment size:', stateSize)
    print('\tOutput/Action size:', acSize)
    print('')
    print('***************************************')
    
    #create architecture
    bot = Agent(acSize, stateSize)
    #plot_model(bot, to_file='model.png', show_shapes=True, show_layer_names=True)

    #train architecture
    bot, scores, times = train(game, bot, renderTraining)

    for i in range(10):
        playFinalGame(game, bot)

    #Print results
    plt.plot(times, scores)
    plotTitle = game + ' Training Scores'
    plt.title(plotTitle)
    plt.xlabel('iterations')
    plt.ylabel('scores')
    plt.show()
    # bot.loadNN('./save/nnWeights.h5')



