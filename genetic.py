import math
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import numpy as np


# 计算适应度
def compute_fitness(pop_line,distance):
    sum_dis = 0
    temp_dis = 0
    nums = len(pop_line)
    for i in range(nums):
        if i < nums - 1:
            temp_dis = distance[pop_line[i],pop_line[i+1]]
            sum_dis += temp_dis
        else:
            temp_dis = distance[pop_line[i],pop_line[0]]
            sum_dis += temp_dis
    return sum_dis

# 轮盘赌选择（不太好实现，先不用了）
# def select(pops,fitness):
#     fitness = np.array(fitness)
#     index = np.random.choice(np.arange(POP_SIZE), size=1, replace=True,p=( fitness / fitness.sum()))[0]
#     print(fitness[index])
#     return pops[index]

# 自然选择，采用锦标赛选择
def tournament_select(pops,pop_size,fitness,tournament_size):
    new_pops = []
    new_fitness = []
    # 直到新的种群规模到达当前种群规模
    while len(new_pops) < len(pops):
        # 从原始样本中选出的样本
        checked_list_pop = [random.randrange(0,pop_size) for i in range(0,tournament_size)]
        checked_list_fitness = np.array([fitness[i] for i in checked_list_pop])
        min_fitness = checked_list_fitness.min() # 最小适应度
        idx = np.where(checked_list_fitness == min_fitness)[0][0] # 获取索引
        min_pop = pops[idx] #获取对应的个体
        new_pops.append(min_pop) #放入新的种群中
        new_fitness.append(min_fitness)
    # print(new_pops)
    return new_pops #返回新的种群

# 交叉算子 选择单点交叉
def crossover(pop_size,pops_parent1,pops_parent2,trans_rate):
    pops_children = []
    # 双亲交叉一次得到一个子代
    for i in range(0,pop_size):
        child = []
        parent1 = pops_parent1[i]
        parent2 = pops_parent2[i]
        # print("双亲为:")
        # print(parent1)
        # print(parent2)
        if random.random() > trans_rate:
            # 不发生交叉，则随机保留父代中的一个
            if random.random() > 0.5:
                child = parent1.copy()
            else:
                child = parent2.copy()
        else:
            #发生交叉
            #确定交叉的位置
            child = parent2.copy()
            begin = random.randrange(0,len(parent1))
            end = random.randrange(0,len(parent1))
            if begin > end:
                temp = begin
                begin = end
                end = temp
            #print("从%d到%d" %(begin,end))
            for i in range(begin,end+1):
                child[i] = parent1[i]
            #print("子代为:",child)
        pops_children.append(child)
    return pops_children

#变异算子 采用交换突变
def mutate(pop_size,pops,mutation_rate):
    pops_mutated = []
    for i in range(pop_size):
        child = pops[i].copy()
        if random.random() < mutation_rate:
            # 发生变异
            pos_first = random.randrange(0,len(child))
            pos_second = random.randrange(0,len(child))
            if pos_first != pos_second:
                temp = child[pos_first]
                child[pos_first] = child[pos_second]
                child[pos_second] = temp
        pops_mutated.append(child)
    return pops_mutated

# elitism策略处理（精英注意）
def elitism(pop_size,pops,children_pops,pop_fitness,children_fitness):
    #如果父代适应度更高的话，则不会被淘汰
    for i in range(0,pop_size):
        if pop_fitness[i] > children_fitness[i]:
            pops[i] = children_pops[i]
            pop_fitness[i] = children_fitness[i]
    return pops,pop_fitness
if __name__ == '__main__':

    CITY_NUM = 10  # 城市数量
    ITERATIONS = 80  # 终止条件的选择 种群繁衍80代终止
    POP_SIZE = 50  # 种群大小
    TOURNAMENT_SIZE = 2  # 锦标赛采样大小
    TRANS_RATE = 0.8  # 交叉概率
    MUTATION_RATE = 0.1  # 变异概率

    # 获取城市坐标
    coordinates = [] # 城市坐标
    with open("data.txt","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        coordinates.append(list(map(int,line.split(" "))))
    coordinates = np.array(coordinates)
    row,col = coordinates.shape

    #计算各个城市之间的距离
    distance = np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            distance[i,j] = distance[j,i] = math.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2 )
    #print(distance)

    iteration = 0 #迭代次数为0

    #生成种群 100个 每一种路径的选择相当于一种染色体
    pops = [random.sample([i for i in list(range(len(coordinates)))],len(coordinates)) for j in range(POP_SIZE)]
    #print(pops)

    #计算适应度
    pop_fitness = [0] * POP_SIZE
    for i in range(POP_SIZE):
        pop_fitness[i] = compute_fitness(pops[i],distance)
    # print(pop_fitness)

    # 找到初代最优解
    min_pop_fitness = min(pop_fitness)
    optimal_pop = pops[pop_fitness.index(min_pop_fitness)]
    print("第1代的最短距离为:%f" % min_pop_fitness)
    optimal_pops = [optimal_pop]

    start_time = time.perf_counter()
    while iteration <ITERATIONS:
        pops_parent1 = tournament_select(pops,POP_SIZE,pop_fitness,TOURNAMENT_SIZE)
        pops_parent2 = tournament_select(pops,POP_SIZE,pop_fitness,TOURNAMENT_SIZE)

        pops_children = crossover(POP_SIZE,pops_parent1,pops_parent2,TRANS_RATE)
        # print(pops_children)

        pops_mutated = mutate(POP_SIZE,pops_children,MUTATION_RATE)
        # print(pops_mutated)

        # 统计发生变异的个体
        # print( "发生变异的个体有%d个"  %(len(np.where(np.array(pops_children) != np.array(pops_mutated))[0]) / 2) )

        children_fitness = []
        for i in range(POP_SIZE):
            children_fitness.append(compute_fitness(pops_mutated[i],distance))
        pops,pop_fitness = elitism(POP_SIZE,pops,pops_children,pop_fitness,children_fitness)
        # for i in range(POP_SIZE):
        #     if pop_fitness[i] > children_fitness[i]:
        #         pop_fitness[i] = children_fitness[i]
        #         pops[i] = pops_children[i]
        # 找到当代最优解
        if min_pop_fitness > min(pop_fitness):
            min_pop_fitness = min(pop_fitness)
            optimal_pop = pops[pop_fitness.index(min_pop_fitness)]

        optimal_pops.append(optimal_pop)
        print("第%d代的最短距离为:%f" % (iteration+2,min_pop_fitness))
        iteration += 1
    end_time = time.perf_counter()
    print("迭代所需时间为:%f ms" % ((end_time-start_time)*1000))
