import numpy as np #подключение numpy

'''
1. Объясните, в чём отличие функций arange() и linspace()?
при использовании arange может произойти потеря точности


2. Что такое срез массива?
Подмножество массива
 
3. Допустим, m — это двумерный массив. Что означают кон-струкции m [:,   j] и  m[i ,   :]?
m [:, j] вывод j-го столбца
m[i , :] вывод i-го столбца

4. Что означают отрицательные индексы в срезах массивов?
Если они относятся к столбцам или строкам, то это означает отсчитывать элемент с конца
Если же в шаге, то обратный ход

5. Как быстро определить номера минимальных и максималь-ных элементов в данном массиве?
c помощью функци argmax argmin

6. Что такое универсальные функции?
Оболочки для обычных математических функций, выполняющие обработку массивов типа ndarray поэлементно. 
Скомпилированы они на языке С, а значит исполняются быстрее, чем стандартный for на Python
'''
#разное создание массивов
#способ 1
my_list = [0, 1, 2, 3, 4]
a1 = np.array(my_list)
print(a1)
print(type(a1))

#способ 2
a2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
print(a2)

#способ 3
a3 = np.arange(5, dtype=float)
#другие параметры
a4 = np.arange(11, 16, 2, dtype=float)

#способ 4
a5 = np.linspace(0, 1, 6, dtype=float)

#способ 5 только yekb
a6 = np.zeros(7, dtype=int)

#способ 6 только единицы
a7 = np.ones(3, dtype=int)

#доступ к элементам
a1 = np.array([1, 4, 5, 8])
print(a1[3])
a1[0] = 5.
print(a1)

a2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
a2[0, 1]

#пример срезов
a1 = np.linspace(0, 2, 9)
print(a1)
#каждый с 2го по 4й
print(a1[1:5:1])
print(a1[1:3])
#все элементы
print(a1[::])
print(a1[:])
#каждый второй
print(a1[::2])
print(a1[4:])
#c 3 с конца
print(a1[-3::])
print(a1[:-3:])
print(a1[::-1])

#общий вид
#a2[start_row:stop_row:step_row, start_column:stop_column:step_column]

a2 = np.array([[-1, 0, 2, 4], [10, 20, 40, 80], [7, 7, 7, 7]])

#Нулевой столбец массива a2:
print(a2[:,0])

#Новый массив
new_array = a2[1, :] + a2[2, :]
print(new_array)

#Использование некоторых функций
a = np.array([2, 4, 3], float)
#сумма всех элементов массива
print(a.sum)
#произведение всех элементов массива
print(a.prod())

#среднее арифметическое всех элементов массива
print(a.mean())

print(a.min())
print(a.max())

#получение мин и макс элементов массива

print(a.argmin())
print(a.argmax())

print(a.size)
#размерность
print(a.ndim)
print(a.shape)

#Универсальные функции
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
#поэлементно
print(a+b)
print(a-b)
print(a**b)
#матричное произведение
print(np.dot(a,b))

x = np.linspace(0, 9, 10)
print(np.sin(x))

#работа с файлом
#numpy.loadtxt(fname, dtype=<type ‘float’>, comments=’#’, delimiter=None, skiprows=0, usecols=None)
