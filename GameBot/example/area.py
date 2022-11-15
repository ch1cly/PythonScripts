# Создадим среду и извлечем пространство действий
env = Cure()
np.random.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Построим модель актера. Подаем среду, получаем действие
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(4, use_bias=True))
actor.add(Activation('relu'))
actor.add(Dense(4, use_bias=True))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions, use_bias=True))
actor.add(Activation('tanh'))
print(actor.summary())

# Построим модель критика. Подаем среду и действие, получаем награду
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(8, use_bias=False)(x)
x = Activation('relu')(x)
x = Dense(5, use_bias=True)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Keras-RL предоставляет нам класс, rl.memory.SequentialMemory
# где хранится "опыт" агента:
memory = SequentialMemory(limit=100000, window_length=1)
# чтобы не застрять с локальном минимуме, действия модели полезно "встряхивать" случайным поведением
# с помощью Процесса Орнштейна – Уленбека
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
# Создаем agent из класса DDPGAgent
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)

agent.compile(Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# Обучим процесс на nb_steps шагах,
# nb_max_episode_steps ограничивает количество шагов в одном эпизоде
agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=Epochs)

# Тестируем обученую сеть на 5 эпизодах
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=Epochs)
env.close()