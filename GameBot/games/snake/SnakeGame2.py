# %%
import time
import random
from IPython.display import clear_output
import numpy as np
import keyboard
#from freegames import square, vector
#import turtle as t
import pygame as p

# %%
def actionAsk():
    a = np.zeros(4,dtype=np.int64)
    if keyboard.is_pressed('Down'):
        a[2] = 1
    if keyboard.is_pressed('Left'):
        a[1] = 1
    if keyboard.is_pressed('Up'):
        a[0] = 1
    if keyboard.is_pressed('Right'):
        a[3] = 1
    return a


# %%
class Point2():
    def __init__(self, x=0, y=0):
        self._info = np.array([x, y], ndmin=1)

    @property
    def x(self):
        return self._info[0]

    @property
    def y(self):
        return self._info[1]

    @x.setter
    def x(self, a):
        self._info[0] = a

    @y.setter
    def y(self, a):
        self._info[1] = a

    def __add__(self, o):
        return Point2(self._info[0] + o._info[0], self._info[1] + o._info[1])

    def __sub__(self, o):
        return Point2(self._info[0] - o._info[0], self._info[1] - o._info[1])

    def __eq__(self, other):
        if np.all(np.equal(self._info, other._info)):
            return True
        else:
            return False

    def norm(self):
        l = np.sqrt(self.x*self.x+self.y*self.y)
        self.x = self.x/l
        self.y = self.y/l


# %%
class CustomEnv():
    def __init__(self, h=10, w=10):
        self.WIDTH = int(w)
        self.HEIGHT = int(h)
        self.snake = []
        self.food = Point2(0, 0)
        self.aim = Point2(0, -1)
        self.endGame = False
        self.memory = 0
        self.startLen = 0
        self.Time = True
        self.factor = 20
        self.display = p.display.set_mode((640, 640))
        '''
        t.title("Snake")
        root = t.Screen()._root
        # root.iconbitmap("logo-ico.ico")
        t.bgcolor('#99ffbb')
        t.setup((self.HEIGHT + 3) *  self.factor, (self.WIDTH + 3) * self.factor, 600, 0)
        t.hideturtle()
        t.tracer(False)
'''
    def inside(self, head):
        return -1 < head.x < self.WIDTH and -1 < head.y < self.HEIGHT

    def chooseD(self,action)->Point2:
        p = Point2(0,0)
        if action == 0:
            p.y = -1
        elif action == 1:
            p.x = -1
        elif action == 2:
            p.y = 1
        elif action == 3:
            p.x = 1

        return p

    def change(self, a):
        if np.sum(a) == 1:
            action = np.argmax(a)
            if np.abs(self.memory - action) != 2 and action != -1:
                self.aim = self.chooseD(action)
                self.memory = action

    def step(self, action):
        self.steps += 1
        self.change(action)
        head = self.snake[-1] + self.aim
        reward = 0
        if (not self.inside(head) or head in self.snake) and self.aim != Point2(0, 0) or self.Time and self.steps > (self.WIDTH+self.HEIGHT*2) * (len(self.snake)):
            self.endGame = True
            reward = -10
            done = True
        else:
            done = False
            self.snake.append(head)

            if head == self.food:
                reward = 10
                if len(self.snake) != self.dim1():
                    self.generateFood()
                else:
                    done = True
                    self.endGame = True
            else:
                self.snake.pop(0)

        info = len(self.snake) - self.startLen
        return reward, done, info

    def render(self, episode=0, reward=0):
        clear_output(wait=True)
        s = ''
        for i in range(self.WIDTH, -2,  - 1):
            for j in range(self.HEIGHT, -2,  -1):
                if Point2(i, j) == self.snake[-1]:
                    s = s + '0'
                elif Point2(i, j) in self.snake:
                    s = s + '█'
                elif Point2(i, j) == self.food:
                    s = s + 'X'
                elif j == -1 or j == self.HEIGHT:
                    s = s + '‖'
                elif i == -1 or i == self.WIDTH:
                    s = s + '='
                else:
                    s = s + '·'
            s = s + '\n'
        s = s + 'Episode:{} Score:{} Snake:{}'.format(episode, reward, len(self.snake))
        print(s)

    def position(self, pos) -> int:
        return pos.x * self.HEIGHT + pos.y

    def generateFood(self):
        self.food = self.rPoint()
        while self.food in self.snake:
            self.food = self.rPoint()

    def snakePos(self):
        obsspc = np.zeros(self.HEIGHT * self.WIDTH, dtype=np.float64)
        for i in range(len(self.snake)):
            obsspc[self.position(self.snake[i])] = 0.33
        obsspc[self.position(self.snake[-1])] = 0.66
        obsspc[self.position(self.food)] = 0.99
        return obsspc

    def rPoint(self) -> Point2:
        x = random.randint(0, self.WIDTH - 1)
        y = random.randint(0, self.HEIGHT - 1)
        return Point2(x, y)

    def reset(self, seed=2, playerSeed=False, tttime = True):
        if (playerSeed):
            random.seed(seed)
        else:
            t = 1000 * time.time()  # current time in milliseconds
            random.seed(int(t) % 2 ** 32)
        self.snake = [self.rPoint()]#[Point2(self.WIDTH//2,self.HEIGHT//2),Point2(self.WIDTH//2-1,self.HEIGHT//2),Point2(self.WIDTH//2-2,self.HEIGHT//2)]
        self.generateFood()
        self.steps = 0
        self.aim = Point2(0, 0)
        self.endGame = False
        self.startLen = len(self.snake)
        self.memory = -20
        self.Time = tttime
        return self.snakePos()

    def dim1(self) -> int:
        return self.WIDTH * self.HEIGHT

    def dim2(self)->int:
        return 16

    def getState1(self):
        return self.snakePos()

    def iscollision(self,p):
        if not self.inside(p):
            return True
        if p in self.snake:
            return True
        return False

    def isThereSnake(self,aim):
        if len(self.snake)>1:
            p = self.chooseD(aim%4)
            for i in range(len(self.snake)-1):
                q = self.snake[i] - self.snake[-1]
                q.norm()
                if q==p:
                    return True
        return False

    def getState2(self):
        return np.array([
            self.iscollision(self.snake[-1] + Point2(0,-1)),
            self.iscollision(self.snake[-1] + Point2(-1,0)),
            self.iscollision(self.snake[-1] + Point2(0,1)),
            self.iscollision(self.snake[-1] + Point2(1,0)),


            # Move direction
            self.memory == 0,
            self.memory == 1,
            self.memory == 2,
            self.memory == 3,

            self.isThereSnake(0),
            self.isThereSnake(1),
            self.isThereSnake(2),
            self.isThereSnake(3),

            # Food location

            self.food.y < self.snake[-1].y,  # food down
            self.food.x < self.snake[-1].x,  # food left
            self.food.y > self.snake[-1].y,  # food up
            self.food.x > self.snake[-1].x  # food right
        ])

    def actionInput(self):
        return 4

    def getScore(self):
        return len(self.snake)
    def renderHuman(self):
        GREEN = (0,255,0)
        BLACK = (0,0,0)
        RED = (255,0,0)
        YELLOW = (255,255,0)
        BROWN = (204,102,0)
        BLUE = (0,0,255)
        s = self.factor
        self.display.fill(GREEN)
        p.display.set_caption('Score: '+str(len(self.snake)))

        shifty = self.HEIGHT * s
        shiftx = self.WIDTH * s
        for i in range(-1, self.WIDTH+1):
            p.draw.rect(self.display, BLACK, p.Rect(-i * s + shiftx + s, 0, s, s))
            p.draw.rect(self.display, BLACK, p.Rect(-i * s + shiftx + s, shifty + s, s, s))

        for i in range(-1, self.HEIGHT+1):
            p.draw.rect(self.display, BLACK, p.Rect(0    ,  -i * s + shifty+ s, s, s))
            p.draw.rect(self.display, BLACK, p.Rect(shiftx+ s, -i * s + shifty+ s, s, s))

        p.draw.rect(self.display, RED, p.Rect(self.food.x*s+ s, self.food.y*s+ s, s, s))

        for pt in self.snake:
            p.draw.rect(self.display, YELLOW, p.Rect(pt.x*s+ s, pt.y*s+ s, s, s))



        if self.endGame:
            p.draw.rect(self.display, BLUE, p.Rect(self.snake[-1].x * s+ s, self.snake[-1].y * s+ s, s, s))
        else:
            p.draw.rect(self.display, BROWN, p.Rect(self.snake[-1].x * s+ s, self.snake[-1].y * s+ s, s, s))

        p.display.flip()
'''
    def renderHuman(self, episode=0, reward=0):
        factor = self.factor
        t.clear()
        shifty = -self.HEIGHT//2 * factor
        shiftx = -self.WIDTH// 2 * factor
        for i in range(0, self.WIDTH):
            square(i * factor + shiftx, self.HEIGHT * factor + shifty , factor, '#000000')
            square(i * factor + shiftx, -1 * factor + shifty            , factor , '#000000')

        for i in range(0, self.HEIGHT):
            square(self.WIDTH * factor + shiftx, i * factor+ shifty , factor, '#000000')
            square(-1 * factor + shiftx        , i * factor+ shifty , factor, '#000000')

        square(self.food.x * factor + shiftx, self.food.y * factor+ shifty , factor, '#cc99ff')

        for s in self.snake:
            square(s.x * factor + shiftx, s.y * factor+ shifty , factor, 'brown')

        if self.endGame:
            square(self.snake[-1].x * factor + shiftx, self.snake[-1].y * factor+ shifty , factor, 'red')
        square(self.snake[-1].x * factor + shiftx, self.snake[-1].y * factor+ shifty , factor, 'yellow')

        t.update()
'''

