# This game is from the official documentation of freegames
# https://pypi.org/project/freegames/

# pip install freegames

# Tap on Tile to Move

# import modules
from random import *
import turtle as t
from freegames import square, vector
from time import sleep

# Set window title, color and icon
WIDTH = HEIGHT = 10
factor = 9
EXITf = False
RESTART = False

def DrawWalls():
    for i in range(-WIDTH,WIDTH):
        square(i*factor, HEIGHT*factor , factor, '#000000')
        square(i*factor,-HEIGHT*factor , factor, '#000000')

    for i in range(-HEIGHT,HEIGHT):
        square(WIDTH*factor, i*factor, factor, '#000000')
        square(-WIDTH*factor, i*factor, factor, '#000000')




t.title("Snake")
root = t.Screen()._root
#root.iconbitmap("logo-ico.ico")
t.bgcolor('#99ffbb')


food = vector(0, 0)
snake = [vector(0, 0)]
aim = vector(0, -1)

def StartGame():
    global food
    global snake
    global aim
    food = vector(0, 0)
    snake = [vector(0, 0)]
    aim = vector(0, -1)



def EndGame():
    global EXITf
    EXITf = True

def Restart():
    global RESTART
    RESTART = True

#   Functions
# Change snake direction
def change(x, y):
    global aim
    aim.x = x
    aim.y = y

# Return True if head inside boundaries
def inside(head):
    return -WIDTH < head.x < WIDTH and -HEIGHT < head.y < HEIGHT

# Move snake forward one segment
def move():
    global EXITf
    global RESTART
    global aim
    global head
    global snake
    head = snake[-1].copy()
    head.move(aim)

    if not inside(head) or head in snake or EXITf or RESTART:
        square(head.x*factor, head.y*factor, factor, 'red')
        t.update()
        RESTART = True
        print("End of the game!\nYour score is {}".format(len(snake)))
        return False

    snake.append(head)

    if head == food:
        print('Snake:', len(snake))
        food.x = randrange(-WIDTH+1, WIDTH)
        food.y = randrange(-HEIGHT+1, HEIGHT)
    else:
        snake.pop(0)

    t.clear()

    for body in snake:
        square(body.x*factor, body.y*factor, factor, '#802b00')
    square(snake[-1].x * factor, snake[-1].y * factor, factor, '#ffcc00')
    DrawWalls()

    square(food.x*factor, food.y*factor, factor, '#cc99ff')
    t.update()
    return True

t.setup((HEIGHT*2+3)*factor, (WIDTH*2+3)*factor, 370, 0)
t.hideturtle()
t.tracer(False)
t.listen()
t.onkey(lambda: change(1, 0), 'Right')
t.onkey(lambda: change(-1, 0), 'Left')
t.onkey(lambda: change(0, 1), 'Up')
t.onkey(lambda: change(0, -1), 'Down')
t.onkey(lambda: EndGame(), 'space')
t.onkey(lambda: Restart(), 'Return')
while True:
    move()
    sleep(0.2)
    print(snake)
    print(food)
    if RESTART:
        StartGame()
        RESTART = False
    if EXITf:
        break
t.bye()