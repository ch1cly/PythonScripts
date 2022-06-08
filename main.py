import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_plot(data):
    l = {2 : 'black',4:'red',8:'purple',16:'green',32:'blue'}
    #plt.title('Ускорение')
    i = 2
    while i < 33:
        #plt.figure(figsize=(14, 6))
        plt.plot(data['j'], data[str(i)], label= 'потоков ' + str(i)
                 ,color=l[i])
        i = i *2
    plt.ylabel('ускорение')
    plt.xlabel('количество точек')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_plot1(data):
    l = {2 : 'black',4:'red',8:'purple',16:'green',32:'blue'}
    plt.title('Зависимость минимального угла от итераций')
    i = 2

    plt.plot(data['i'], data['angle']
                 ,color=l[32])
    plt.hlines(26, 0 ,315,colors='red')
    plt.ylabel('минимальный угол')
    plt.xlabel('кол-во итераций')
    plt.grid(True)
    plt.show()




def crplot():
    data = pd.read_csv('')

    create_plot(data)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    crplot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
