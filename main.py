import numpy as np
from tabulate import tabulate




def scalar(A, eps):
    x = np.array(np.ones(A.shape[0]))
    y = np.array(np.ones(A.shape[0]))
    lambda_=0
    k=0
    while True:
        k += 1
        x_ = np.dot(A, x)
        y_ = np.dot(A.T, y)
        if abs(np.dot(x_, y_) / np.dot(x, y_) - lambda_) < eps:
            lambda_ = np.dot(x_, y_) / np.dot(x, y_)
            break
        lambda_ = np.dot(x_, y_) / np.dot(x, y_)
        x = x_
        y = y_
    return lambda_, k

def stepen(A, eps):
    x = np.array(np.ones(A.shape[0]))
    lambda_=0
    k=0
    while True:
        k += 1
        x_ = np.dot(A, x)
        if abs(x_[0] / x[0] - lambda_) < eps:
            lambda_ = x_[0] / x[0]
            break
        lambda_=x_[0] / x[0]
        x = x_
    return lambda_, k
def iter_form(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + 1 + j + 1 - 1)
    return A
A2 = np.array([[1, 1 / 2,],
                  [1 / 2, 1 / 3]],dtype=float)
A3 = np.array([[1, 1 / 2, 1 / 3, ],
                  [1 / 2, 1 / 3, 1 / 4],
                  [1 / 3, 1 / 4, 1 / 5]],dtype=float)
A4 = np.array([[-500.7, 120.7],
                  [ 890.3, -550.6]],dtype=float)
p_A2=A2
p_A3=A3
p_A4=A4
print(A4)
iter_form(A2)
iter_form(A3)
iter_form(A4)
lambda_accA2 = max(abs(np.linalg.eig(A2)[0]))
lambda_accA3 = max(abs(np.linalg.eig(A3)[0]))
lambda_accA4 = max(abs(np.linalg.eig(A4)[0]))

print(tabulate([[10**(-3),stepen(A4, 10**(-3))[1],scalar(A4, 10**(-3))[1],abs(lambda_accA4 - abs(stepen(A4, 10**(-3))[0])),abs(lambda_accA4 - abs(scalar(A4, 10**(-3))[0]))],
               [10**(-4),stepen(A4, 10**(-4))[1],scalar(A4, 10**(-4))[1],abs(lambda_accA4 - abs(stepen(A4, 10**(-4))[0])),abs(lambda_accA4 - abs(scalar(A4, 10**(-4))[0]))],
               [10**(-5),stepen(A4, 10**(-5))[1],scalar(A4, 10**(-5))[1],abs(lambda_accA4 - abs(stepen(A4, 10**(-5))[0])),abs(lambda_accA4 - abs(scalar(A4, 10**(-5))[0]))],
               [10**(-6),stepen(A4, 10**(-6))[1],scalar(A4, 10**(-6))[1],abs(lambda_accA4 - abs(stepen(A4, 10**(-6))[0])),abs(lambda_accA4 - abs(scalar(A4, 10**(-6))[0]))]], headers=['Погрешность','#Итерации степенного','#Итерации скалярного','|lambda_acc - lambda|(step)','|lambda_acc - lambda|(skal)'],tablefmt='orgtbl'))

print(p_A3)
print(tabulate([[10**(-3),stepen(A3, 10**(-3))[1],scalar(A3, 10**(-3))[1],abs(lambda_accA3 - abs(stepen(A3, 10**(-3))[0])),abs(lambda_accA3 - abs(scalar(A3, 10**(-3))[0]))],
                [10**(-4),stepen(A3, 10**(-4))[1],scalar(A3, 10**(-4))[1],abs(lambda_accA3 - abs(stepen(A3, 10**(-4))[0])),abs(lambda_accA3 - abs(scalar(A3, 10**(-4))[0]))],
                [10**(-5),stepen(A3, 10**(-5))[1],scalar(A3, 10**(-5))[1],abs(lambda_accA3 - abs(stepen(A3, 10**(-5))[0])),abs(lambda_accA3 - abs(scalar(A3, 10**(-5))[0]))],
                [10**(-6),stepen(A3, 10**(-6))[1],scalar(A3, 10**(-6))[1],abs(lambda_accA3 - abs(stepen(A2, 10**(-6))[0])),abs(lambda_accA3 - abs(scalar(A3, 10**(-6))[0]))]], headers=['Погрешность','#Итерации степенного','#Итерации скалярного','|lambda_acc - lambda|(step)','|lambda_acc - lambda|(skal)'],tablefmt='orgtbl'))
print(p_A2)

print(tabulate([[10**(-3),stepen(A2, 10**(-3))[1],scalar(A2, 10**(-3))[1],abs(lambda_accA2 - abs(stepen(A2, 10**(-3))[0])),abs(lambda_accA2 - abs(scalar(A2, 10**(-3))[0]))],
                [10**(-4),stepen(A2, 10**(-4))[1],scalar(A2, 10**(-4))[1],abs(lambda_accA2 - abs(stepen(A2, 10**(-4))[0])),abs(lambda_accA2 - abs(scalar(A2, 10**(-4))[0]))],
                [10**(-5),stepen(A2, 10**(-5))[1],scalar(A2, 10**(-5))[1],abs(lambda_accA2 - abs(stepen(A2, 10**(-5))[0])),abs(lambda_accA2 - abs(scalar(A2, 10**(-5))[0]))],
                [10**(-6),stepen(A2, 10**(-6))[1],scalar(A2, 10**(-6))[1],abs(lambda_accA2 - abs(stepen(A2, 10**(-6))[0])),abs(lambda_accA2 - abs(scalar(A2, 10**(-6))[0]))]], headers=['Погрешность','#Итерации степенного','#Итерации скалярного','|lambda_acc - lambda|(step)','|lambda_acc - lambda|(skal)'],tablefmt='orgtbl'))