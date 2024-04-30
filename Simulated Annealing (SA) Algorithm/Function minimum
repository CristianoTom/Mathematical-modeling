import random
import math
import matplotlib.pyplot as plt
import numpy as np

# 定义目标函数
def function(x):
    return x*math.sin(math.pi*x)

def casual(x):
    return random.uniform(0,18)

# 定义模拟退火算法函数
def sa(function,x,t_initial,alpha,t_min):
    target = x
    t_current = t_initial

    path=[function(x)]

    while t_current > t_min:
        sm=0
        while sm<1:
            newtarget = casual(target)
            difference = function(newtarget) - function(target)
            if difference < 0 or random.uniform(0.4, 1) < math.exp(-difference / t_current):
                target = newtarget
                sm = 0
            else:
                sm += 1
            path.append(function(target))
        t_current *= alpha
        
    return target,path
        

# 设置初始参数
x = 1.0 # 初始解
t_initial = 200.0 # 初始温度
alpha = 0.95 # 冷却率
t_min = 1e-10 # 最低温度

# 调用模拟退火算法求解
x, path = sa(function,x,t_initial,alpha,t_min)


# 输出结果
print("最优解:", x)
print("最优值:", function(x))


#可视化部分

plt.figure(figsize=(10, 6))
plt.plot(path, 'r')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Simulated Annealing Optimization Process')
plt.grid(True)
plt.show()
