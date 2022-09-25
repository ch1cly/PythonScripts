'''
1. Опишите иерархию объектов на рисунке в Matplotlib.
> Все объекты имеет иерархическую структуру
Основным является рисунок, панели и оси координат
Сверху иерархии находится рисунок, который служит контейнером для остальных объектов - класс Figure
На рисунке распологаютс панели - класс Axes, на панелях содержатся разнообразные элементы рисунков



2. Пусть fig — объект-рисунок, созданный функцией figure(). По- ясните результат работы следующих команд:
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
Как будут располагаться панели на рисунке fig?
> создается таблица из двух строк и одной колонки, в ней ax1 занимает первую позицию, а ax2 вторую

3. Ниже приведены команды рисования графиков зависимостей y1(x), y2(x), y3(x). Какими линиями будут нарисованы графики 1, 2, и 3?
plt.plot(x, y1, ‘k-’, label=’1’)
plt.plot(x, y2, ‘g:’ , label=’2’)
plt.plot(x, y3, ‘--’, color=’orange’ , label=’3’)
>   линия один - черная ломанная
    линия два - зеленые точки
    линия три - орнажевая прирывистая

4. В чём отличие функций contour() и contourf()?
> пространство между контурами заполняется заливкой

5. Пусть функция f определена в некоторой области на плоско- сти (x, y) и имеет в этой области минимальное значение 0,
мак- симальное значение 10. Чем будет отличаться вывод команд contour(x, y, f, levels=[0, 5, 10]) и contour(x, y, f, 10)?
>Отображается разное число изолиний, в первом случае 3, во втором 10

6. Воспользуйтесь функцией help() и выясните, для чего предна- значена функция matplotlib.pyplot.imshow().
>выводит изображение
'''

def f1():

    # подключение модуля pyplot под псевдонимом plt
    import matplotlib.pyplot as plt

    # подключение библиотеки numpy под псевдонимом np
    import numpy as np

    # matplotlib.pyplot.plot(x, y, args) - общий вид функции


    # массив координат - 30 точек, равнораспределенных # в диапазоне от 0 до 10
    x = np.linspace(0.0, 10.0, 30)
    # массив значений функции в заданных координатах
    y = np.sin(x)
    # рисование графика функции с помощью функции plot
    plt.plot(x, y)
    # отображение рисунка на экране
    plt.show()

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

def f3():
    # подключение библиотеки numpy и модуля pyplot
    import numpy as np
    import matplotlib.pyplot as plt
    # создание рисунка размером 15 на 15 см # с одной панелью
    inch = 2.54  # дюйм в см
    fig1 = plt.figure(figsize=(15.0 / inch, 15.0 / inch))
    ax1 = fig1.add_subplot(111)
    # подписи осей на панели
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    # массив x-координат - 50 точек в диапазоне от -2 до 2
    x = np.linspace(-2.0, 2.0, 50)
    # массив y-координат - 50 точек в диапазоне от -2 до 2
    y = np.linspace(-2.0, 2.0, 50)
    # матрицы (сетка) координат
    xx, yy = np.meshgrid(x, y)
    # вычисление значений функции на сетке
    F = np.exp(-xx ** 2 - yy ** 2)
    # отображение 10 изолиний величины F.
    # график изолиний ассоциируется с переменной CS1
    CS1 = ax1.contour(xx, yy, F, 10)
    # добавление подписей изолиний на графике CS1.
    # с помощью обращения к полю levels (списку изолиний # на графике CS1) подписи выводятся только
    # для каждой второй линии
    ax1.clabel(CS1, CS1.levels[::2])
    fig1.show()

def f4():
    # подключение библиотеки numpy и модуля pyplot
    import numpy as np
    import matplotlib.pyplot as plt
    # создание рисунка размером 15 на 15 см # с одной панелью
    inch = 2.54  # дюйм в см
    fig1 = plt.figure(figsize=(18.0 / inch, 15.0 / inch))
    ax1 = fig1.add_subplot(111)
    # подписи осей на панели
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    # массив x-координат - 50 точек в диапазоне от -2 до 2
    x = np.linspace(-2.0, 2.0, 50)
    # массив y-координат - 50 точек в диапазоне от -2 до 2
    y = np.linspace(-2.0, 2.0, 50)
    # матрицы (сетка) координат
    xx, yy = np.meshgrid(x, y)

    # функция для отображения
    def F(x, y):
        return np.exp(-x ** 2 - y ** 2)

    # градиент функции
    def gradF(x, y):

        # x-координата градиента
        gradF_x = -2 * x * F(x, y)
        # y-координата градиента
        gradF_y = -2 * y * F(x, y)
        return [gradF_x, gradF_y]
    # вычисление значений функции на сетке
    FF = F(xx, yy)
    # отображение 20 изолиний величины F с заливкой.
    # график изолиний ассоциируется с переменной CS
    # аргумент cmap указывает используемую цветовую схему
    CS = ax1.contourf(xx, yy, FF, 20, cmap='Blues')
    # добавление на рисунок легенды,
    # указывающей соответствие цветов заливки
    # уровням величины F на графике CS fig1.colorbar(CS, label=r’$F(x,y)$’)
    # вычисление градиента функции на сетке
    grad = gradF(xx, yy)
    # рисование поля градиента с помощью векторов # векторное поле ассоциируется с переменной q
    q = ax1.quiver(xx, yy, grad[0], grad[1])
    # отображение легенды для векторного поля - # стрелка длиной 1 для масштаба
    ax1.quiverkey(q, X=0.5, Y=1.05, U=1, label=r'$ | \vec{\nabla} F |= 1$', labelpos ='E')
    fig1.show()


def f5():
    from matplotlib import pyplot as plt
    import numpy as np
    data = np.loadtxt("data.txt")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(u'Данные из файла')
    # рисование графика зависимости первого столбца от #нулевого
    ax.plot(data[:, 0], data[:, 1], 'ko -', color ='grey', label = r'$d_1$')
    # рисование графика зависимости второго столбца от #нулевого
    ax.plot(data[:, 0], data[:, 2], 'gs -', label = r'$d_2$')
    ax.legend(loc='best')
    fig.show()