
# Имитация роевого поведения
class Colony:
    # положения частицы
    x : np.ndarray
    y : np.ndarray
    # угол направления частицы
    theta : np.ndarray
    # скорость частицы по осям
    vx : np.ndarray
    vy : np.ndarray

    # Конструктор
    def __init__(self,N):
        self.reset(N)

    # расстановка N частиц на площадке LxL
    def reset(self,N):
        # положения частиц
        self.x = np.random.rand(N,1)*L
        self.y = np.random.rand(N,1)*L
        # направление и осевые скорости частиц относительно
        # постоянной линейной скорости v0
        self.theta = 2 * np.pi * np.random.rand(N,1)
        self.vx = v0 * np.cos(self.theta)
        self.vy = v0 * np.sin(self.theta)
    # Шаг имитации
    def step(self):
        # движение
        self.x += self.vx*dt
        self.y += self.vy*dt
        # применение периодических пограничных условий
        self.x = self.x % L
        self.y = self.y % L
        # найти средний угол соседей в диапазоне R
        mean_theta = self.theta
        for b in range(N):
            neighbors = (self.x-self.x[b])**2+(self.y-self.y[b])**2 < R**2
            sx = np.sum(np.cos(self.theta[neighbors]))
            sy = np.sum(np.sin(self.theta[neighbors]))
            mean_theta[b] = np.arctan2(sy, sx)
        # добавление случайного отклонения
        self.theta = mean_theta + eta*(np.random.rand(N,1)-0.5)
        # изменение скорости
        self.vx = v0 * np.cos(self.theta)
        self.vy = v0 * np.sin(self.theta)
        return self.theta

    # Получить список частиц в внутри радиуса r от координат x,y
    def observe(self,x,y,r):
        return (self.x-x)**2+(self.y-y)**2 < r**2
    # Вывести координаты частицы i
    def print(self,i):
        return print(self.x[i],self.y[i])
    # Получить координаты частиц
    def get_bacteria(self):
        return self.x, self.y
    # Получить массив направлений частиц
    def get_theta(self):
        return self.theta



