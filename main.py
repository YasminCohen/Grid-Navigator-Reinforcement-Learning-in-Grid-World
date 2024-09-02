import numpy as np
import pandas as pd
import MDP,baseModelRL,modelFree

df = pd.read_excel(r"C:\Users\97253\Downloads\gridResults.xlsx")

i = 0
j = 12

def run(w,h,L,p,r):
    global i,j



    MDPValues, MDPPolicy = MDP.calculate_optimal_policy(w,h,L,p,r)
    BmValues, BmPolicy = baseModelRL.model_based_rl(w,h,L,p,r)
    FrValues,FrPolicy = modelFree.calculate_optimal_policy(w,h,L,p,r)

    diff12 = abs(np.mean(MDPValues-BmValues))
    diff13 = abs(np.mean(MDPValues-FrValues))
    diff23 = abs(np.mean(BmValues-FrValues))


    df.loc[i,'average(d(MDP, MBRL))'] = diff12
    df.loc[i,'average(d(MDP,MFRL))'] = diff13
    df.loc[i,'avrage(d(MBRL,MFRL)) '] = diff23

    t = 2
    for y in range(h):
        for x in range(w):
            df.iloc[j,t] = abs(MDPValues[y,x]-BmValues[y,x])
            df.iloc[j+1,t] = abs(MDPValues[y,x]-FrValues[y,x])
            df.iloc[j+2,t] = abs(BmValues[y,x]-FrValues[y,x])
            t += 1
    j += 5



    print(diff12)
    print(diff13)
    print(diff23)
    i += 1

    df.set_index("test")

#Location coordinates starts at 0. lower left is (0,0), upper right is (w-1,h-1).
#t1 grid from first MDP lecture
w = 4
h = 3
L = [(1,1,0),(3,2,1),(3,1,-1)]
p = 0.8
r = -0.04

run(w,h,L,p,r)

#t2 grid from first MDP lecture
w = 4
h = 3
L = [(1,1,0),(3,2,1),(3,1,-1)]
p = 0.8
r = 0.04

run(w,h,L,p,r)



#t3 grid from first MDP lecture
w = 4
h = 3
L = [(1,1,0),(3,2,1),(3,1,-1)]
p = 0.8
r = -1

run(w,h,L,p,r)

#t4 the cliff from last RL lecture

w = 12
h = 4
L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,1)]
p = 1
r = -1

run(w,h,L,p,r)

#t5 the cliff2

w = 12
h = 6
L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,1)]
p = 0.9
r = -1

run(w,h,L,p,r)

#t6

w = 5
h = 5
L = [(4,0,-10),(0,4,-10),(1,1,1),(3,3,2)]
p = 0.9
r = -0.5

run(w,h,L,p,r)

#t7

w = 5
h = 5
L = [(2,2,-2),(4,4,-1),(1,1,1),(3,3,2)]
p = 0.9
r = -0.25

run(w,h,L,p,r)

#t8

w = 7
h = 7
L = [(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
p = 0.8
r = -0.5

run(w,h,L,p,r)

#t9

w = 7
h = 7
L = [(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
p = 0.8
r = -0.5

run(w,h,L,p,r)

#t10

w = 7
h = 7
L = [(3,1,0),(3,5,0),(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
p = 0.8
r = -0.25

run(w,h,L,p,r)


df.to_excel(r"C:\Users\97253\Downloads\gridResults.xlsx")
