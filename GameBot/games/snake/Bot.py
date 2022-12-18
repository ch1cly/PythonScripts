import random
import numpy as np
from random import randint as rd

'''
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
            self.food.y > self.snake[-1].y, # food up
            self.food.x > self.snake[-1].x  # food right
            
'''


def randomBool()->bool:
    return rd(0, 20) == 11

def randomBool1()->bool:
    return rd(0, 1) != 1

def predictState16(state):
    action = np.zeros(4,dtype=np.int32)

    if np.sum(state[4:9]) == 0:
        for i in range(3):
            if state[i] == 0 and state[12+i] ==1:
                action[i] = 1
                return action


    lurSnake = np.array([0,0,0],dtype=np.int32)
    lurFood = np.array([0, 0, 0], dtype=np.int32)
    lur = np.array([0,0,0],dtype=np.int32)

    lurCollision = np.array([0,0,0],dtype=np.int32)
    lur[1] = np.argmax(state[4:8]) #forward
    lur[0] = (lur[1]-1)%4 #left
    lur[2] = (lur[1]+1)%4 #right
    #заполню значения
    for i in range(3):
        lurSnake[i] = state[8+lur[i]]
        lurCollision[i] = state[lur[i]]
        lurFood[i] = state[12+lur[i]]


    for i in range(3):
        if lurSnake[i] == 0 and lurFood[i] == 1 and lurCollision[i] == 0:
            action[lur[i]] = 1
            return action


    if np.sum(lurSnake)==3:
        for i in range(3):
            if lurCollision[i] == 0:
                action[lur[i]] = 1
                return action



    if np.sum(lurSnake) == 0 and np.sum(lurFood) == 0 and lurCollision[1] == 1:
        if lurCollision[0]==1:
            action[lur[2]] = 1
        else:
            action[lur[0]] = 1
        return action

    '''если варианты есть'''

    if np.sum(state[12:]) == 1 and (state[12] == 1 or state[15] == 1) and \
            (state[8] == 1 or state[11]==1) and np.sum(state[8:12])==2:
        if state[2] == 0:
            action[2] = 1
        elif state[0] == 0:
            action[0] = 1

        return action

    for i in range(3):
        if lurSnake[i-1] == 1 and lurSnake[i] == 1:
            if lurCollision[lur[i-2]] == 0: #можно еще в другую сторону
                action[lur[i-2]] = 1
            else :
                if rd(0,10) % 2 == 0:
                    action[lur[i-1]] = 1
                    action[lur[i]] = 0
                    action[lur[i-2]] = 0
                else:
                    action[lur[i - 1]] = 0
                    action[lur[i]] = 1
                    action[lur[i - 2]] = 0
            return action

    for i in range(10):
        i = rd(0, 2)
        if lurSnake[i] == 0 and lurCollision[i] == 0:
            action[lur[i]] = 1
            return action

    return action


if __name__ == '__main__':
    import time

    t = 1000 * time.time()  # current time in milliseconds
    random.seed(int(t) % 2 ** 32)
    state = [
        True,  True, False,  True, False, False,  True, False,  True,  True, False, False,
        True,  True, False, False
    ]
    predictState16(state)