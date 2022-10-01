'''
Библиотека scipy предоставляет широкий набор инструментов для эффективных научных вычислений
Она содержит все частоприменяемые компоненты по типу: констант, методов, алгоритмов и тп

'''

def f1():
    from scipy import integrate
    help(integrate.quad) #вызов справки для функции
    '''
    Вывод(справка по функции quad):
    Integrate func from `a` to `b` (possibly infinite interval) using a
    technique from the Fortran library QUADPACK. 
    ...и так далее
    '''


def f2():
    '''
    quad (func, a, b, ...) - наидолее простая функция вычисления определенных интегралов
    пример использования
    '''
    from scipy import integrate
    def f(x):
        return 3.0 * x ** 2

    print(integrate.quad(f, 0.0, 4.0))
    print()

def f3():
    '''
    пример вычисления int{0}{inf} (exp^-x)dx
    '''
    import numpy as np
    from scipy import integrate
    def f02(x):
        return np.exp(-x)

    print(integrate.quad(f02, 0.0, np.inf))
    '''
    также c помощью scipy можно решать ОДУ 1го порядка
    '''

if __name__ == '__main__':
    f1()
    f2()
    f3()