import gym
import numpy as np
import random
from IPython.display import clear_output

# Инициализируем Taxi-V2 Env
env = gym.make("Taxi-v3").env

# Инициализируем произвольные значения
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Гиперпараметры
alpha = 0.1
gamma = 0.6
epsilon = 0.1


all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    # Инициализируем переменные
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            # Проверяем пространство действий
            action = env.action_space.sample()
        else:
            # Проверяем изученные значения
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Обновляем новое значение
        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: {i}")

print("Training finished.")