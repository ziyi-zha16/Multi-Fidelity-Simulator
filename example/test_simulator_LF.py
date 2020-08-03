import os,sys       #这几行（1~3）在安装了之后可以省略
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 

from MF_Sim.LF_Sim import simulator

import numpy as np
import time


# a=simulator.Agent(pos_x=2,pos_y=7,goal_x=27,goal_y=7,direct=0,vel = 1)
# b=simulator.Agent(pos_x=4,pos_y=5,goal_x=15,goal_y=5,direct=0,vel = 1)
# c=simulator.Agent(pos_x=5,pos_y=1,goal_x=5,goal_y=10,direct=1,vel = 1)
# agents=[]
# agents.append(a)
# agents.append(b)
# agents.append(c)

# map=np.zeros((30,30))
# map[0,:] = 1
# map[:,0] = 1
# map[30-1,:] = 1
# map[:,30-1] = 1
# map[22,0:15]=1
# w = simulator.Full_env(30, 30, False,3,agents,map)

w =simulator.Full_env(30, 30, True,3)


# w = simulator.Full_env(30, 30, False,3)
w.reset()
w.random_fill(30,30,room_number=5)


# w.set_start([[2,7],[16,7],[5,1]])
# w.set_goal([[20,7],[20,5],[5,10]])
# w.set_direct([0,0,1])


for i in range(100000) :
    #time.sleep(0.1)
    w.render(0)
    if ((i+1)%100)==0:
        #w.reset(True)
        w.reset()
        #print('w.map[0][0]',w.map[0][0],'w.map[1][1]',w.map[1][1])
    # print('方向1：',w.agents[0].direct)
    # print('原始方向1：',w.agents[0].direct_origin)
    #print('x坐标：',w.agents[0].pos.x)
    #print('y坐标：',w.agents[0].pos.y)

    action=[]
    for a in range(3):
        act=w.action_space[0].sample()
        action.append(act)
    #print(action)
    obs, reward, done,info = w.step(action)

    # obs, reward, done,info = w.step([2,2,2])
    #print('obs',obs)
    print('done',done)
    print('reward:',reward)
    print('info',info)
    # print('方向：',w.agents[0].direct)
    # print('原始方向：',w.agents[0].direct_origin)
    # print('x坐标：',w.agents[0].pos.x)
    # print('y坐标：',w.agents[0].pos.y)
