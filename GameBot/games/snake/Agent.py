import time
from games.snake.SnakeGame2 import CustomEnv, actionAsk
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
import torch
from game import SnakeGameAI
from game1 import SnakeGameAI as sa
from Bot import predictState16
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self,env,variant=2):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.env = env
        self.reset = self.env.reset
        self.step = self.env.step
        if variant==1:
            self.get_state = self.env.getState1
            self.dim = self.env.dim1
            self.model = Linear_QNet(self.dim(), 1024, self.env.actionInput())
        elif variant==2:
            self.get_state = self.env.getState2
            self.dim = self.env.dim2
            self.model = Linear_QNet(self.dim(), 256, self.env.actionInput())
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = int(self.env.dim1()*0.5) - self.n_games
        fmove = np.zeros(shape=self.env.actionInput(),dtype=np.int32)
        if random.randint(0, self.env.dim1()*0.5) < self.epsilon:
            move = random.randint(0, self.env.actionInput()-1)
            fmove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            fmove[move] = 1

        return fmove

    def humanPlay(self):
        self.env.reset()
        while True:  # Run until solved
            self.env.renderHuman()  # Adding this line would show the attempts
            time.sleep(0.1)
            state = self.get_state()
            action = predictState16(state)
            reward, done, _ = self.env.step(actionAsk())
            print(self.get_state())
            if done:
                break

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        self.reset()
        while record < 200:
            # get old state
            state_old = self.get_state()
            # get move
            final_move = self.get_action(state_old)


            # perform move and get new state
            reward, done, score = self.step(final_move)
            #self.env.renderHuman()
            state_new = self.get_state()
            # train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            self.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                self.reset()
                self.n_games += 1
                self.train_long_memory()

                if score > record:
                    record = score
                    self.model.save()

                print('Game', self.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = np.array(total_score / self.n_games, dtype=np.float32)
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


    def test(self):
        self.model.load()
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        self.reset()
        #self.env.renderHuman()
        while True:
            # get old state

            state = self.get_state()
            # get move
            move = self.get_action(state)

            # perform move and get new state
            reward, done, score = self.step(move)
            # train short memory
            #self.env.renderHuman()
            time.sleep(0.01)
            if done:
                state = self.reset()
                print('score is ', score)

    def botGame(self):
        self.env.reset(tttime=False)
        while True:  # Run until solved
            state = self.get_state()
            print(state)
            action = predictState16(state)
            self.env.renderHuman()  # Adding this line would show the attempts
            time.sleep(0.01)
            reward, done, _ = self.step(action)
            if done:
                break
                self.env.reset(tttime=False)

if __name__ == '__main__':
    #a = Agent(CustomEnv(30,30))
    #a.train()
    a = Agent(variant=1,env=CustomEnv(30,30))
    a.train()