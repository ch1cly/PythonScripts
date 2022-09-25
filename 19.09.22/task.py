'''
Модифицируйте программу рисования графиков, приведённую в параграфе 3.3, в соответствии со следующими требованиями:
1. Уберите заголовок рисунка, сетку и легенду.
2. Нарисуйте график каждой из четырёх функций в отдельной панели на рисунке. Оси каждой панели должны быть подписаны.
На каждой панели сверху по центру следует добавить текстовую метку: «(а)» для панели 1, «(б)» для панели 2, «(в)» для панели 3, «(г)» для панели 4.
3. График синуса должен быть нарисован сплошной чёрной ли- нией, график косинуса — пунктирной чёрной линией,
график квадрата косинуса — штриховой чёрной линией, график функции x0.15 — сплошной серой линией.
4. Установите в каждой панели диапазон значений по оси аб- сцисс от 0 до 2π.
Диапазон значений по оси ординат на каждой па- нели должен быть таким, чтобы график соответствующей функ- ции был виден целиком.
'''

import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt

def f2():
    # подключение библиотеки numpy под псевдонимом np
    import numpy as np
    # подключение модуля pyplot из библиотеки matplotlib # под псевдонимом plt
    import matplotlib.pyplot as plt
    # пользовательская переменная для хранения размера шрифта
    fsize = 12
    # настройка типа шрифта на рисунке с помощью изменения # записей в словаре rcParams из модуля pyplot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # настройка размера шрифта в различных частях рисунка # в заголовке:
    plt.rcParams['axes.titlesize'] = fsize
    # в подписях осей:
    plt.rcParams['axes.labelsize'] = fsize
    # в подписях меток на осях:
    plt.rcParams['xtick.labelsize'] = fsize
    plt.rcParams['ytick.labelsize'] = fsize
    # в легенде рисунка:
    plt.rcParams['legend.fontsize'] = fsize
    # массив координат – 50 точек в диапазоне [0, 10]
    x = np.linspace(0.0, 10.0, 50)
    # создаём окно рисунка.
    # Для дальнейшей работы рисунок ассоциируется с переменной # fig
    fig = plt.figure()
    # добавляем панель (оси координат) с именем ax в окно fig. # в дальнейшем настройка осей производится
    # через обращение к переменной ax.
    # аргументы 1, 1, 1 указывают, что на рисунке будет
    # только одна панель для рисования графиков
    ax = fig.add_subplot(1, 1, 1)
    # график синуса:
    # кружки (o), соединённые сплошной (-) чёрной линией. # графику присваивается строковый идентификатор ‹1› # для дальнейшего отображения в легенде
    ax.plot(x, np.sin(x), 'ko -', label ='1')
    # график косинуса:
    # квадратики (s, размером 3), соединённые сплошной (-) # оранжевой линией толщиной 1.
    # графику присваивается строковый идентификатор ‹2›
    # для отображения в легенде
    ax.plot(x, np.cos(x), 'ks -', color ='orange', linewidth = 1, markersize = 3.0, label ='2')
    # график синуса в квадрате:
    # треугольники (^), соединённые сплошной (-) лиловой # линией толщиной 1.
    # графику присваивается строковый идентификатор ‹3› # для отображения в легенде
    ax.plot(x, (np.sin(x)) ** 2.0, 'k ^ -', color ='magenta', linewidth = 1, label ='3')
    # график функции f(x)=x^0.15:
    # чёрная штриховая линия толщиной 1.
    # графику присваивается строковый идентификатор ‹x^2’ # для отображения в легенде.
    # символ r и знаки доллара внутри строки позволяют
    # вводить математические символы с помощью команд ТеХ
    ax.plot(x, (x) ** 0.15, 'k - -', linewidth = 1, label = r'$x ^ 2$')
    # легенда
    ax.legend(loc='best')
    # диапазон отображаемых значений по оси х
    ax.set_xlim(-1.0, 11.0)
    # диапазон отображаемых значений по оси y

    ax.set_ylim(-1.5, 1.5)
    # подпись по оси x
    ax.set_xlabel(r'$x$')
    # подпись по оси y
    ax.set_ylabel(r'$f(x)$')
    # заголовок рисунка
    ax.set_title('Мой первый рисунок')
    # сетка на рисунке
    ax.grid()
    # сохраняем в файл с именем fig1 типа PNG с разрешением # 300 точек на дюйм
    # (dpi – dots per inch), с альбомной ориентацией
    fig.savefig("fig1.png", orientation='landscape', dpi = 300)