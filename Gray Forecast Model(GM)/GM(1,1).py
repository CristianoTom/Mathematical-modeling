import numpy as np
import pandas as pd
#import sympy as sp

#1.级比lambda检验
x0 = [30,68,75,82]     #共有n个元素
x00= [i for i in x0]
x0 = np.array(x0,dtype = float)
standard = [np.exp(-2 / (len(x0) + 1)) , np.exp(2 / (len(x0) + 1))]
lamb = [x0[i] / x0[i+1] for i in range(len(x0)-1)]

#定义判断函数
def t(x) :
    if standard[0] < x < standard[1]:
        return 1
    return 0

j = 0
while sum([t(i) for i in lamb]) != len(x0) - 1 :
    x00= [i + j for i in x0]
    lamb = [x00[i] / x00[i+1] for i in range(len(x0)-1)]
    j += 1

j *= 1.5
j = int(j)    #平移常数
print(j)


#2.构建GM（1,1）模型
x1 = [sum([x0[j] for j in range(0 , i)]) for i in range(1 , len(x0)+1)] 
B = np.array([[- (x1[i] + x1[i+1]) / 2 , 1] for i in range(len(x0) - 1)])
Y = np.array([x0[i] for i in range(1,len(x0))])

#求解参数矩阵
target = np.matmul(np.linalg.inv(np.matmul(B.T,B)),np.matmul(B.T,Y))
a,b = target[0], target[1]
print(a,b)

#给出拟合方程
def f(x):
    return (x0[0] - b / a)*np.exp(- a * x) + b / a

#预测函数差分出的预测值
ls = [f(i) - f(i - 1) for i in range(1,len(x0))]
ls.insert(0,int(x0[0]))
ls = np.array(ls)


# 输出结果
test_table_columns = ['Year', 'Raw Data', 'Predction', 'Error', 'Relative_Error', '级比偏差']
test_table = pd.DataFrame(np.c_[[str(i) for i in range(1,5)],
                                x0,
                                ls,
                                x0 - ls,
                                np.divide((x0 - ls), x0),
                                [np.nan] + [1-(1-0.5*a)/(1+0.5*a)*lamb[k] for k in range(len(lamb))]],
                          columns=test_table_columns)
test_table['Predction'] = test_table['Predction'].apply(lambda x: format(float(x), '.4f'))
test_table['Error'] = test_table['Error'].apply(lambda x: format(float(x), '.4f'))
test_table['Relative_Error'] = test_table['Relative_Error'].apply(lambda x: format(float(x), '.2%'))
test_table['级比偏差'] = test_table['级比偏差'].apply(lambda x: format(float(x), '.4f'))
print('\n', test_table, sep='')
