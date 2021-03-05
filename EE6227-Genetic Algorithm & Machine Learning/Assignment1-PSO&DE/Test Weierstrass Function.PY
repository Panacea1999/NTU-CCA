#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: Weierstrass Function.PY
@Abstract: Evaluate performance of PSO, DE/rand/1, DE/best/1, DE/CTB/1 based on Weierstrass Function
@Time: 2021/03/06 00:54:08
@Requirements: numpy, geatpy, pandas, matplotlib
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import pandas as pd

rand1_recorder = []
best1_recorder = []
ctbest_recorder = []
pso_recorder = []
rand1_avg_recorder = []
rand1_best_recorder = []
best1_avg_recorder = []
best1_best_recorder = []
ctbest_avg_recorder = []
ctbest_best_recorder = []
pso_best_recorder = []

class MyProblem(ea.Problem):  # Define the Problem
    def __init__(self):
        name = 'Weierstrass Function'  # Initialize name
        M = 1  # Dimension of Objective
        maxormins = [1]  # 1: Maximum, 2: Minimum
        Dim = 10  # Dimension of Decision Variables
        varTypes = [0] * Dim  # 1: Discrete, 0: Continuous
        lb = [-0.5]*Dim  # Low Bound
        ub = [0.5]*Dim  # Up Bound
        lbin = [1]*Dim
        ubin = lbin
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # Objective Function
        x = pop.Phen  # Decision Variable Matrix
        sum1 = np.zeros_like(x)
        sum2 = 0
        for k in range(20):
            sum1 = sum1 + 0.5**(k+1)*np.cos(2*np.pi*(x+0.5)*3**(k+1))
            sum2 = sum2 + 0.5**(k+1)*np.cos(2*np.pi*3**(k+1)*0.5)
        pop.ObjV = np.sum((sum1), 1, keepdims=True)-10*sum2

from PSO import PSOKW
#Objective Function for PSO
def objective(x):
    sum1 = np.zeros_like(x)
    sum2 = 0
    for k in range(20):
        sum1 = sum1 + 0.5**(k+1)*np.cos(2*np.pi*(x+0.5)*3**(k+1))
        sum2 = sum2 + 0.5**(k+1)*np.cos(2*np.pi*3**(k+1)*0.5)
    res = np.sum((sum1), 1, keepdims=True)-10*sum2

    return res

if __name__ == '__main__':
    #Generate Problem Objective
    problem = MyProblem()  # 生成问题对象
    #Set Parameter
    runtime = 10
    Encoding = 'RI'  # Coding
    NIND = 100  # NP
    MAXGEN = 500 # Max Generation
    F = 0.6 #Parameter
    XOVR = 0.7 #Parameter
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Region Describer
    population = ea.Population(Encoding, Field, NIND)  # Instantiate the Population

#Instantiate the Algorithm
##############################DE/rand/1#################################
    myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAXGEN
    myAlgorithm.mutOper.F = F
    myAlgorithm.recOper.XOVR = XOVR
    myAlgorithm.logTras = 0  # Interval of Generation for Log
    myAlgorithm.verbose = True  # Print Log
    myAlgorithm.drawing = 0  # Dont Draw

    #Call Algorithm and Run
    for i in range(runtime):
        [BestIndi, population] = myAlgorithm.run()  # Get the Best Individual and the Latest Population
        BestIndi.save()
        print('Evaluate：%s Iterations' % myAlgorithm.evalsNum)
        print('Time used: %s Second' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('Run Time: %s' %(i+1))
            print('Best Fitness Value of DE/rand/1:' % BestIndi.ObjV[0][0])
            rand1_recorder.append(BestIndi.ObjV[0][0])
            print('Corresponding Decision Variables')
            for i in range(BestIndi.Phen.shape[1]):
                print("%2.7f"%BestIndi.Phen[0, i])
        else:
            print('No Feasible Solution')
        rand1_best_recorder.append(myAlgorithm.trace['f_best'])
        rand1_avg_recorder.append(myAlgorithm.trace['f_avg'])

##############################DE/best/1#################################
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAXGEN
    myAlgorithm.mutOper.F = F
    myAlgorithm.recOper.XOVR = XOVR
    myAlgorithm.logTras = 0
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0
    #Call Algorithm and Run
    for i in range(runtime):
        [BestIndi, population] = myAlgorithm.run()
        BestIndi.save()
        print('Evaluate：%s Iterations' % myAlgorithm.evalsNum)
        print('Time used: %s Second' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('Run Time: %s' %(i+1))
            print('Best Fitness Value of DE/best/1：%s' % BestIndi.ObjV[0][0])
            best1_recorder.append(BestIndi.ObjV[0][0])
            print('Corresponding Decision Variables')
            for i in range(BestIndi.Phen.shape[1]):
                print("%2.7f"%BestIndi.Phen[0, i])
        else:
            print('No Feasible Solution')
        best1_best_recorder.append(myAlgorithm.trace['f_best'])
        best1_avg_recorder.append(myAlgorithm.trace['f_avg'])

##############################DE/currentToBest/1#################################
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAXGEN
    myAlgorithm.mutOper.F = F
    myAlgorithm.recOper.XOVR = XOVR
    myAlgorithm.logTras = 0
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0
    #Call Algorithm and Run
    for i in range(runtime):
        [BestIndi, population] = myAlgorithm.run()
        BestIndi.save()
        print('Evaluate：%s Iterations' % myAlgorithm.evalsNum)
        print('Time used: %s Second' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('Run Time: %s' %(i+1))
            print('Best Fitness value of DE/currentToBest/1:' % BestIndi.ObjV[0][0])
            ctbest_recorder.append(BestIndi.ObjV[0][0])
            print('Corresponding Decision Variables')
            for i in range(BestIndi.Phen.shape[1]):
                print("%2.7f"%BestIndi.Phen[0, i])
        else:
            print('No Feasible Solution')
        ctbest_best_recorder.append(myAlgorithm.trace['f_best'])
        ctbest_avg_recorder.append(myAlgorithm.trace['f_avg'])

#####################################PSO####################################



#Call Algorithm and Run
for i in range(runtime):
    pso = PSOKW(dim=10, fitness=objective, constraints=[], population_size=NIND, max_iter=MAXGEN)
    pso.c1 = 1.5
    pso.c2 = 3.5
    pso.init_Population(low=-0.5, high=0.5) # Bound of Union of xi
    pso.solve()
    fit = pso.fit
    # obj fitness
    x = pso.gbest.reshape((1,-1))
    obj_fit = objective(x)
    pso_recorder.append(obj_fit[0])
    pso_best_recorder.append(pso.record_fit)
    print('Run Time: %s' %(i+1))
    print('Best Fitness value of PSO：%s' % obj_fit[0])

#Save Data
data = [{'mean_DE1':np.mean(rand1_recorder),"median_DE1":np.median(rand1_recorder),"std_DE1":np.std(rand1_recorder),"recorder_DE1":rand1_recorder,\
    "recorder_best1":min(rand1_recorder),'mean_DE2':np.mean(best1_recorder),"median_DE2":np.median(best1_recorder),"std_DE2":np.std(best1_recorder),"recorder_DE2":best1_recorder,\
    "recorder_best2":min(best1_recorder),'mean_DE3':np.mean(ctbest_recorder),"median_DE3":np.median(ctbest_recorder),"std_DE3":np.std(ctbest_recorder),"recorder_DE3":ctbest_recorder,\
    "recorder_best3":min(ctbest_recorder),'mean_pso':np.mean(pso_recorder),"median_pso":np.median(pso_recorder),"std_pso":np.std(pso_recorder),"recorder_pso":list(pso_recorder),\
    "recorder_pso_best":min(pso_recorder)}]
pd.DataFrame(data).to_csv(r'C:\Users\wangzy\Desktop\EE6227\assignment1\TestFunction5.csv')

all_recorder = rand1_recorder + best1_recorder + ctbest_recorder + pso_recorder
best_index = all_recorder.index(min(all_recorder))
all_iter_recorder = rand1_best_recorder + best1_best_recorder + ctbest_best_recorder + pso_best_recorder
all_avg_recorder = rand1_avg_recorder + best1_avg_recorder + ctbest_avg_recorder

#Log Process
'''
rand1_recorder = np.log(rand1_recorder)
best1_recorder = np.log(best1_recorder)
ctbest_recorder = np.log(ctbest_recorder)
pso_recorder = np.log(pso_recorder)
'''



#Draw
plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title('Different Results under Multiple Runs')
plt.grid(linestyle='-.')
plt.plot(np.arange(1,len(rand1_recorder)+1,1),rand1_recorder, linewidth=1,linestyle='-',marker='.', color='darkcyan',label='DE')
plt.plot(np.arange(1,len(rand1_recorder)+1,1),best1_recorder, linewidth=1,linestyle='-' ,marker='.',color='darkred',label='DE/best/1')
plt.plot(np.arange(1,len(rand1_recorder)+1,1),ctbest_recorder, linewidth=1,linestyle='-' ,marker='.',color='darkblue',label='DE/CurrentToBest/1')
plt.plot(np.arange(1,len(rand1_recorder)+1,1),pso_recorder, linewidth=1,linestyle='-',marker='.', color='darkgreen',label='PSO')
plt.legend(fontsize=12)

plt.xticks([1,5,10,15,20,25,30])
plt.xlabel('Run times', fontsize=12)
plt.ylabel('Fitness Value of Objective Function', fontsize=12)
#plt.ylabel('Logarithmic Fitness Value of Objective Function', fontsize=12)

plt.subplot(122)
plt.title('Optimization Process for\nBest Result among Multiple Runs')
plt.grid(linestyle='-.')
plt.plot(np.arange(1,len(all_iter_recorder[0])+1,1),(all_iter_recorder[0]), linewidth=1, color='darkcyan', label='Best Fitness')
plt.plot(all_avg_recorder[0], linewidth=1, color='red', label='Average Fitness')
plt.xlabel('Number of Generation', fontsize=12)
#plt.ylabel('Logarithmic Fitness Value of Objective Function', fontsize=12)
plt.ylabel('Fitness Value of Objective Function', fontsize=12)
plt.legend(fontsize=12)
plt.suptitle('Weierstrass Function:\nBast Known Result = 0, Best Result by this Algorithm Set = '+str(all_recorder[best_index]), fontsize=14)
plt.show()
print("mean:",np.mean(rand1_recorder))
print("median:",np.median(rand1_recorder))
print("std:",np.std(rand1_recorder))
print(len(x_recorder))

print(all_recorder)

