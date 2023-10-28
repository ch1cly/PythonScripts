import numpy as np
import random
import copy


# Function description
def f(x):
    return 2 + 4 * x + 9 * (x ** 2) + 7 * (x ** 3)


def p_i(size, x):
    if x == size:
        return 2
    return 1 + x ** 3


def q(x):
    return x + 1


def a_i(size, i, h):
    return p_i(size, i * h - h) + p_i(size, i * h) / 2


class matrix:
    # initialisation forming np.arrays and set dimension
    def __init__(self, size=0, low=0, high=0):
        self.matr = np.random.uniform(low, high, (size, size))
        self.b = np.random.uniform(low, high, size)
        self.x = np.zeros(size)
        self.xtest = np.zeros(size)
        self.low = low
        self.high = high
        self.size = size

        while np.linalg.det(self.matr) == 0 and size != 0 and low * high != 0:
            self.matr = np.random.uniform(low, high, (size, size))

    # form another matr
    def refresh(self):

        self.matr = np.random.uniform(self.low, self.high, (self.matr.shape[0], self.matr.shape[0]))

        while np.linalg.det(self.matr) == 0 and self.low * self.high != 0:
            self.matr = np.random.uniform(self.low, self.high, (self.matr.shape[0], self.matr.shape[0]))

    def isSymmetricAndPositive(self):
        r = True
        for i in range(1, len(self.matr) + 1):
            if np.linalg.det(self.matr[:i, :i]) <= 0:
                print('Not positive')
                r = False

        for i in range(self.size):
            for j in range(self.size):
                if self.matr[i][j] != self.matr[j][i]:
                    print('Not symmet')
                    print(self.matr[i][j])
                    print(self.matr[j][i])
                    r = False

        return r

    def consolePrint(self,iter):
        print("номер итерации ", iter)
        p = 0
        for a in self.x:
            print('  ' ,a)
            p = p + 1


    def getSimmetricalMatr(self):
        self.formSimmetricalMatr()
        while (not self.isSymmetricAndPositive()):
            print('not formed!')
            self.formSimmetricalMatr()

    def formSimmetricalMatr(self):

        for i in range(self.size):
            for j in range(i + 1):
                self.matr[i][j] = random.uniform(self.low, self.high)
                self.matr[j][i] = self.matr[i][j]

        self.getMatrWithDiagSuperior()



        self.matr = np.dot(self.matr, self.matr)

        for i in range(self.size):
            for j in range(i):
                self.matr[j][i] = self.matr[i][j]

    def getMatrWithDiagSuperior(self):
        self.formDiagDominant()

        while not self.diagDominantCheck() or np.linalg.det(self.matr) == 0:
            self.refresh()
            self.formDiagDominant()

    def formDiagDominant(self):

        for i in range(0, self.matr.shape[0]):
            a = 0
            for j in range(0, self.matr.shape[0]):
                a = a + abs(self.matr[i][j])

            a = a - abs(self.matr[i][i])

            if abs(self.matr[i][i]) < abs(a):
                q = abs(a) - abs(self.matr[i][i])
                qwe = np.sign(self.matr[i][i]) * (q + 1)
                self.matr[i][i] = self.matr[i][i] + qwe

    def diagDominantCheck(self):

        for i in range(0, self.matr.shape[0]):
            a = 0
            for j in range(0, self.matr.shape[0]):
                a = a + abs(self.matr[i][j])

            a = a - abs(self.matr[i][i])

            if abs(a) / abs(self.matr[i][i]) >= 1:
                print('error not diag')
                return False

        # print('ok')
        return True

    def getError(self, x1=np.array(0), flag=False):
        if flag:
            x1 = np.linalg.solve(self.matr, self.b)

        return max(abs(x1 - self.x))

    def yakobi(self, eps=0.00001, k=1000000):
        x1 = np.ones(self.size)
        self.x = np.zeros(self.size)
        er = self.getError(x1)
        k1 = 0
        while er > eps and k1 < k:
            self.x = x1.copy()
            for i in range(self.size):
                # x1[i] = self.b[i] / self.matr[i][i] - sum([self.matr[i][j] * self.x[j] / self.matr[i][i] if i != j else 0 for j in range(0, len(self.b))])

                sumlow = 0
                sumup = 0
                for j in range(0, i):
                    sumlow = sumlow + (self.x[j] * self.matr[i][j])

                for j in range(i + 1, self.size):
                    sumup = sumup + (self.x[j] * self.matr[i][j])

                x1[i] = (- sumlow - sumup + self.b[i]) / self.matr[i][i]

            er = self.getError(x1)
            k1 = k1 + 1


        self.x = x1.copy()

        return k1, er

    def zeidel(self, eps=0.000001, k=1000000):
        x1 = np.ones(self.size)
        er = self.getError(x1)
        k1 = 0
        while er > eps and k1 < k:
            self.x = x1.copy()
            for i in range(self.size):
                # x1[i] = self.b[i] / self.matr[i][i] - sum([self.matr[i][j] * self.x[j] / self.matr[i][i] if i != j else 0 for j in range(0, len(self.b))])

                sumlow = 0
                sumup = 0
                for j in range(0, i):
                    sumlow = sumlow + (x1[j] * self.matr[i][j])

                for j in range(i + 1, self.size):
                    sumup = sumup + (self.x[j] * self.matr[i][j])

                x1[i] = (- sumlow - sumup + self.b[i]) / self.matr[i][i]

            er = self.getError(x1)
            k1 = k1 + 1
            #if(k1 % 10 == 0):
                #self.consolePrint(k1)


        self.x = x1.copy()

        return k1, er

    # dorming matrix for exercise
    def zadan(self, size=0):
        self.size = size
        h = 1 / self.size
        M = 1
        self.matr = np.eye(self.size)
        self.b = np.zeros(size)

        self.matr[0][0] = (a_i(self.size, 1, h) + a_i(self.size, 2, h) + (h ** 2) * q(1))
        self.b[0] = f(h) * (h ** 2)

        for i in range(1, self.size - 1):
            self.matr[i][i - 1] = a_i(self.size, i + 1, h)
            self.matr[i][i] = (a_i(self.size, i + 1, h) + a_i(self.size, i + 2, h) + (h ** 2) * q(i + 1))
            self.matr[i][i + 1] = a_i(self.size, i + 2, h)
            self.b[i] = f(i + 1) * (h ** 2)

        self.matr[self.size - 1][self.size - 2] = -a_i(self.size, self.size, h)
        self.matr[self.size - 1][self.size - 1] = (
                    a_i(self.size, self.size, h) + (h ** 2) * q(self.size) / 2 + h * p_i(self.size, self.size))
        self.b[self.size - 1] = (h ** 2) * f(self.size) / 2 - h * p_i(self.size, self.size) * M

        self.b = np.delete(self.b, self.size - 1, 0)
        self.b = np.delete(self.b,0,0)

        self.matr = np.delete(self.matr, self.size - 1, 0)
        self.matr = np.delete(self.matr, self.size - 1, 1)
        self.matr = np.delete(self.matr,0,0)
        self.matr = np.delete(self.matr, 0, 1)

        self.x = np.zeros(size - 2)
        self.size = self.size - 2

        qweqew=1

    # form b for test
    def test(self, x):
        self.xtest = x
        self.b = self.matr.dot(self.xtest)

    def xtestx(self):
        return max(abs(self.xtest - self.x))

    def relaxation(self, w=1, eps=0.000001, k=1000000):
        x1 = np.ones(self.size)
        er = self.getError(x1)
        k1 = 0
        while er > eps and k1 < k:
            self.x = x1.copy()
            for i in range(self.size):
                # x1[i] = self.b[i] / self.matr[i][i] - sum([self.matr[i][j] * self.x[j] / self.matr[i][i] if i != j else 0 for j in range(0, len(self.b))])

                sumlow = 0
                sumup = 0
                for j in range(0, i):
                    sumlow = sumlow + (x1[j] * self.matr[i][j])

                for j in range(i + 1, self.size):
                    sumup = sumup + (self.x[j] * self.matr[i][j])

                x1[i] = ((1 - w) * self.x[i]) + (w * (- sumlow - sumup + self.b[i])) / self.matr[i][i]

            # print(k1)
            er = self.getError(x1)
            k1 = k1 + 1

            #if (k1 % 5 == 0):
                #self.consolePrint(k1)

        self.x = x1.copy()

        return k1, er

    def residual(self, Ax=np.array(0), b=np.array(0)):
        return Ax - b

    def tauu(self, rk=np.array(0), A=np.array(0)):
        temp = A.dot(rk)
        return rk.dot(rk) / temp.dot(rk)

    def steepestDescent(self, w=1, eps=0.000001, k=1000000):
        x1 = np.ones(self.size)
        er = self.getError(x1)
        k1 = 0
        while er > eps and k1 < k:
            self.x = x1.copy()
            rk = self.residual(self.matr.dot(self.x), self.b)
            tau = self.tauu(rk, self.matr)

            x1 = self.x - tau * rk

            # print(k1)
            er = self.getError(x1)
            k1 = k1 + 1
            #if (k1 % 20 == 0):
                #self.consolePrint(k1)

        self.x = x1.copy()

        return k1, er

    def tridiagonal_algorithm(self):

        # Прямой ход
        v = np.zeros(self.size)
        u = np.zeros(self.size)
        # для первой 0-й строки
        v[0] = self.matr[0][1] / (-self.matr[0][0])
        u[0] = (- self.b[0]) / (-self.matr[0][0])
        for i in range(1, self.size - 1):  # заполняем за исключением 1-й и (n-1)-й строк матрицы
            v[i] = self.matr[i][i + 1] / (-self.matr[i][i] - self.matr[i][i - 1] * v[i - 1])
            u[i] = (self.matr[i][i - 1] * u[i - 1] - self.b[i]) / (-self.matr[i][i] - self.matr[i][i - 1] * v[i - 1])
        # для последней (n-1)-й строки
        v[self.size - 1] = 0
        u[self.size - 1] = (self.matr[self.size - 1][self.size - 2] * u[self.size - 2] - self.b[self.size - 1]) /\
                           (-self.matr[self.size - 1][self.size - 1] - self.matr[self.size - 1][self.size - 2] * v[self.size - 2])

        #print('Прогоночные коэффициенты v: ', 'v', v)
        #print('Прогоночные коэффициенты u: ', 'u', u)

        # Обратный ход
        self.x[self.size - 1] = u[self.size - 1]
        for i in range(self.size - 1, 0, -1):
            self.x[i - 1] = v[i - 1] * self.x[i] + u[i - 1]

        return self.x
