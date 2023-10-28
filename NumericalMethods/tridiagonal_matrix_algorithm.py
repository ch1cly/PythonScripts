import matrix as m
import numpy as np
import matplotlib.pyplot as plt


def eps_tr():
    n = []
    n_s = []
    for i in range(10,101,5):
        n.append(i)
        n_s.append(2*i)

    plt.title("Графики кол-ва требуемых итераций для решения\nСЛАУ методом прогонки")
    plt.xlabel("размерность уравнений")
    plt.ylabel("количество итераций")
    name = ["Прогонки", "Якоби", "Зейделя", "Наискорейшего спуска", "Релаксации"]
    color = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple', 6: 'yellow', 5: 'indigo', 4: 'orange'}
    plt.plot(n, n_s, color=color[2])

def eps_plot():
    n = []

    for i in range(10, 101, 5):
        n.append(i)

    eps = []
    tridiagonal_algorithm = []
    for i in n:
        q = m.matrix()
        q.zadan(i + 2)
        q1 = q.matr
        b1 = q.b
        x = np.linalg.solve(q1, b1)
        q.tridiagonal_algorithm()
        eps.append(max(abs(q.x - x)))
        tridiagonal_algorithm.append(2 * i)
    print('+')

    plt.title("Графики зависимости погрешности от размера СЛАУ")
    plt.xlabel("размерность уравнений")
    plt.ylabel("Погрешность")
    color = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple', 6: 'yellow', 5: 'indigo', 4: 'orange'}
    plt.plot(n, eps, color=color[2])

    plt.show()

def eps_search():
    print("\nТестирование прогонки")
    n = []

    for i in range(10,61,10):
        n.append(i)

    eps  = []
    tridiagonal_algorithm = []
    for i in n:
        q = m.matrix()
        q.zadan(i+2)
        q1 = q.matr
        b1 = q.b
        x = np.linalg.solve(q1, b1)
        q.tridiagonal_algorithm()
        eps.append(max(abs(q.x - x)))
        tridiagonal_algorithm.append(2*i)
    print('+')

    print('eps')
    for i in eps:
        print(i)

    yakobi = []
    print('+')
    for i in range(3):
        q = m.matrix()
        q.zadan(int((i+1) * 10 + 2))

        ansv = q.yakobi(eps=eps[int(i)])
        yakobi.append(ansv[0])
    print('yakobi')
    for i in yakobi:
        print(i)

    zeidel = []
    for i in n:
        q = m.matrix()
        q.zadan(i + 2)
        q1 = q.matr
        b1 = q.b
        #x = np.linalg.solve(q1, b1)
        ansv = q.zeidel(eps=eps[int(i /10 - 1)])
        zeidel.append(ansv[0])
    print('zeidel')
    for i in zeidel:
        print(i)






    relaxation = []
    print('+')
    for i in n:
        q = m.matrix()
        q.zadan(i + 2)
        q1 = q.matr
        b1 = q.b
        #x = np.linalg.solve(q1, b1)
        ansv = q.relaxation(w=1.6,eps=eps[int(i /10 - 1)])
        relaxation.append(ansv[0])
    print('relaxation')
    for i in relaxation:
        print(i)

    steepestDescent = []
    print('+')
    for i in n:
        q = m.matrix()
        q.zadan(i + 2)
        q1 = q.matr
        b1 = q.b
        #x = np.linalg.solve(q1, b1)
        ansv = q.steepestDescent(eps=eps[int(i /10 - 1)])
        steepestDescent.append(ansv[0])

    print('steepestDescent')
    for i in steepestDescent:
        print(i)

    n_s = [tridiagonal_algorithm,zeidel,steepestDescent, relaxation,yakobi]
    plt.title("Графики кол-ва требуемых итераций для достижения\nточности метода прогонки")
    plt.xlabel("размерность матрицы")
    plt.ylabel("количество итераций")
    name = ["Прогонки","Зейделя","Наискорейшего спуска",'Релаксации',"Якоби"]
    color = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple', 6: 'yellow', 5: 'indigo', 4: 'orange'}
    for i in range(0,4):
        q = m.matrix()
        q.zadan(42)
        q.steepestDescent(k=i)
        plt.plot(n,n_s[i], color=color[i], label='метод '+name[i])
    n1 = [10,20,30]
    plt.plot(n1, n_s[4], color=color[4], label='метод ' + name[4])
    plt.legend()
    plt.show()



if __name__ == '__main__':
    eps_search()
