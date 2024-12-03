import numpy as np
import matplotlib.pyplot as plt

'''15.	Формируется матрица F следующим образом: скопировать в нее А 
и  если в Е количество чисел, больших К в четных столбцах больше, чем сумма чисел в нечетных строках, 
то поменять местами С и Е симметрично, иначе В и С поменять местами несимметрично.
 При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, 
 то вычисляется выражение: A*AT – K * FТ, иначе вычисляется выражение (AТ +G-1-F-1)*K, где G-нижняя треугольная матрица, 
 полученная из А. Выводятся по мере формирования А, F и все матричные операции последовательно.'''
def create_submatrixes_random(s):
    matrixB = np.random.randint(-10,10,size=(s,s))
    matrixC = np.random.randint(-10,10,size=(s,s))
    matrixD = np.random.randint(-10,10,size=(s,s))
    matrixE = np.random.randint(-10,10,size=(s,s))
    return matrixB,matrixC,matrixD,matrixE


def create_submatrixes_txt():
    with open('input.txt', 'r') as file:
        lines = file.readlines()
        
    matrices = []
    current_matrix = []

    for line in lines:
        if line.strip():
            current_matrix.append([int(num) for num in line.split()])
        else:
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []

    if current_matrix:
        matrices.append(np.array(current_matrix))

    return matrices

def create_submatrixes_gen(size):
    s = (size,size)
    matrixB = np.full(s,1)
    matrixC = np.full(s,2)
    matrixD = np.full(s,3)
    matrixE = np.full(s,4)
    return matrixB,matrixC,matrixD,matrixE

def create_matrix_from_sub(b,c,d,e):
    m1 = np.concatenate((b,e),axis=1)
    m2 = np.concatenate((c,d), axis=1)
    mA = np.concatenate((m1,m2), axis=0)
    return mA

def search_k(sub,k):
    sum_k = 0
    for i in range(len(sub)):
        if i % 2 == 0:
            for j in range(len(sub[i])):
                sum_k += 1
    return sum_k

def search_nech(sub):
    sum_nech = 0
    for i in range(len(sub)):
        if i % 2 != 0:
            for j in range(len(sub[i])):
                sum_nech += sub[i][j]
    return sum_nech
def create_matrixF(result,b,c,d,e):
    if result == 1:
        m1 = np.concatenate((c,e),axis=1)
        m2 = np.concatenate((b,d),axis=1)
        mF = np.concatenate((m1,m2),axis=0)
        return mF
    else:
        invert_b = np.flip(b)
        sort_e = np.sort(e)
        m1 = np.concatenate((sort_e,invert_b),axis=1)
        m2 = np.concatenate((c,d),axis=1)
        mF = np.concatenate((m1,m2),axis=0)
        return mF

def matrix_expression(res,mA,mF,k):
    if res == 1:
        print("Выполняется решение примера A*At-K*F")
        return (mA * np.transpose(mA)) - (k * mF)
    else:
        print("Выполняется решение примера (At+G-1-F-1)*K ")
        g = np.tril(mA)
        detA = np.linalg.det(mA)
        if detA == 0:
            mF_inv = np.linalg.pinv(mF)
            g_inv = np.linalg.pinv(g)
        else: 
            mF_inv = np.linalg.inv(mF)
            g_inv = np.linalg.inv(g)
        return (np.transpose(mA) + g_inv - mF_inv) * k
    
def plotmat(matrix):
    col_means = np.mean(matrix, axis=0)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(col_means)), col_means)
    plt.title("Среднее значение по столбцам матрицы")
    plt.xlabel("Индекс столбца")
    plt.ylabel("Среднее значение")
    plt.grid(True)
    row_means = np.mean(matrix, axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(row_means)), row_means)
    plt.title("Среднее значение по строкам матрицы")
    plt.xlabel("Индекс строки")
    plt.ylabel("Среднее значение")
    plt.grid(True)
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Значения матрицы')
    plt.title("Тепловая карта матрицы")
    plt.xlabel("Индекс столбца")
    plt.ylabel("Индекс строки")
    plt.show()


while True:
    chooser = int(input("Как создаём матрицу:\n1 - Рандомом\n2 - из файла\n3 - генератором\nВаш выбор: "))
    if chooser == 1:
        size = int(input("Введите размер матрицы: "))
        submatrixB,submatrixC,submatrixD,submatrixE = create_submatrixes_random(size)
        matrixA = create_matrix_from_sub(submatrixB,submatrixC,submatrixD,submatrixE)
        break
    elif chooser == 2:
        submatrixB = create_submatrixes_txt()[0]
        submatrixC = create_submatrixes_txt()[1]
        submatrixD = create_submatrixes_txt()[2]
        submatrixE = create_submatrixes_txt()[3]
        matrixA = create_matrix_from_sub(submatrixB,submatrixC,submatrixD,submatrixE)
        break
    elif chooser == 3:
        size = int(input("Введите размер матрицы: "))
        submatrixB,submatrixC,submatrixD,submatrixE = create_submatrixes_gen(size)
        matrixA = create_matrix_from_sub(submatrixB,submatrixC,submatrixD,submatrixE)
        break

print("Результат")
print(f"Подматрица B:\n{submatrixB}\nПодматрица C:\n{submatrixC}\nПодматрица D:\n{submatrixD}\nПодматрица E:\n{submatrixE}")
print(f"Матрица А:\n{matrixA}")
k = int(input("Введите коэффицент K: "))
result = 1 if search_k(submatrixE,k) > search_nech(submatrixE) else 0
matrixF = create_matrixF(result,submatrixB,submatrixC,submatrixD,submatrixE)
print(f"Матрица F:\n{matrixF}")
det_matrix_A = np.linalg.det(matrixA)
print(f"Определитель матрицы А: {det_matrix_A}")
diag_sum = sum(np.diagonal(matrixF)) + sum(np.fliplr(matrixF).diagonal())
print(f"Сумма диагоналей матрицы F: {diag_sum}")
result = 1 if det_matrix_A > diag_sum else 0
answer = matrix_expression(result,matrixA,matrixF,k)
print(f"Ответ:\n {answer}")
plotmat(matrixF)