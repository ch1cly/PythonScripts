from colony import*

# наша "чашечка Петри"
class Cure(krl.Env):
    # имитируемая колония
    bacteria: Colony
    # положение нано робота
    x: float
    y: float
    theta: float  # направление нано робота
    R: float  # область видимости бактерий нано роботом
    n_bacteria: int  # сохраняем предыдущее значение количества видимых бактерий для rewarda

    # конструктор
    def __init__(self):
        self.bacteria = Colony(N)
        self.reward_range = (-1, 1)  # (-np.inf, np.inf)
        self.action_space = actionSpace()
        self.observation_space = observationSpace()
        self.R = observation_R
        self.reset()

    #  Формирование вектора обзора observation.
    #  То что происходит в области видимости R от робота.
    def observe_area(self):
        # получим список соседей в радиусе R
        observe_bacteria = self.bacteria.observe(self.x, self.y, self.R)
        # получим список соседей в радиусе R*1.5
        observe_far_bacteria = self.bacteria.observe(self.x, self.y, self.R * 1.5)
        observe_far_bacteria = np.array(np.bitwise_and(observe_far_bacteria, np.invert(observe_bacteria)))

        observation = np.zeros(5)
        # подадим количество соседей
        n_bacteria = np.sum(observe_bacteria)
        observation[0] = n_bacteria / 20

        # посчитаем и подадим среднее направлений соседних бактерий
        sx = np.sum(np.cos(self.bacteria.theta[observe_bacteria]))
        sy = np.sum(np.sin(self.bacteria.theta[observe_bacteria]))
        observation[1] = np.arctan2(sy, sx) / np.pi
        # посчитаем и подадим среднее направление от робота до удаленных бактерий
        sx = np.sum(self.bacteria.x[observe_bacteria] - self.x)
        sy = np.sum(self.bacteria.y[observe_bacteria] - self.y)
        observation[2] = np.arctan2(sy, sx) / np.pi
        # посчитаем и подадим среднее направление от робота до удаленных бактерий
        sx = np.sum(self.bacteria.x[observe_far_bacteria] - self.x)
        sy = np.sum(self.bacteria.y[observe_far_bacteria] - self.y)
        observation[3] = np.arctan2(sy, sx) / np.pi
        if n_bacteria:
            observation[4] = self.theta / np.pi  # подадим направление наноробота
        return np.sum(observe_bacteria), observation

    # старт симуляции
    def reset(self):
        self.bacteria.reset(N)
        self.x = .5 * L
        self.y = .5 * L
        self.theta = actionSpace().sample()
        self.n_bacteria, observation = self.observe_area()
        return observation

    # шаг симуляции
    def step(self, action):
        action = action * 3.2  # np.pi
        #  Для экономии времени при попадании на "чистую воду"
        #  просчитываем симуляцию не выпуская ее для обработки сети
        while True:
            # шаг симуляции бактерий
            self.bacteria.step()
            # шаг робота
            self.theta = np.sum(action)  # % (2*np.pi)
            self.x = self.x + dt * v0 * np.cos(self.theta)
            self.y = self.y + dt * v0 * np.sin(self.theta)
            self.x = self.x % L
            self.y = self.y % L
            # осматриваем окружение
            nBacteria, observation = self.observe_area()
            if np.sum(observation) != 0: break
            if self.n_bacteria > 0: break

        delta = nBacteria - self.n_bacteria
        if delta < 0:
            reward = 50 * delta / self.n_bacteria
        elif delta > 0 and self.n_bacteria:
            reward = 1 + delta
        elif nBacteria > 0:
            reward = 1
        elif nBacteria == 0:
            reward = 0
        else:
            reward = nBacteria
        done = nBacteria > N / 7
        self.n_bacteria = nBacteria
        return observation, reward, done, {}

    # получить координаты робота
    def get_position(self):
        return self.x, self.y, self.R

    # получить координаты всех бактерий
    def get_bacteria(self):
        return self.bacteria.get_bacteria()

    # отразить отладочную информацию
    def render(self, mode='human', close=False):
        # print(self.n_bacteria)
        pass

    # завершить симуляцию
    def close(self):
        pass