# Big_Data_Project
Deep Q-Learning (DQN) applied to openAI gym's classic control games

You must import openAI's gym and have python 3.5 or greater installed.
To import openAI's gym please go to the webpage: https://gym.openai.com/docs/#getting-started-with-gym

Once you have your environment setup, you should be able to run bot.py in a terminal with the command:
python bot.py

The program bot.py implements a Deep Q-Learning MLP network. For this, we used an online model and a target model in parallel to learn the optimal policy.
The architecture of the model's are the same and are defined using Keras, a wrapper for tensorflow
Details on the implementation are based off of this paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
The title of the paper is: Human-level control through deep reinforcement learning

The code has many settings, many of them are hyper-parameters to our network.
If you would just like to run our program, the important settings to do so are:
What game you'd like it to learn - This setting can be changed by commenting in or out the 'game' variable
Whether you would like to see it training or not - variable: renderTraining

The program will then train until it is confident that it has a good solution; the conditions of this change based on the game
In the console, the score of each epoch of the game is displayed

Then, the program will play 10 games using the final weights the program determined was an optimal policy
The scores of each of the 10 games will be printed to the console.

Lastly, a plot of the scores across training epochs is displayed.

