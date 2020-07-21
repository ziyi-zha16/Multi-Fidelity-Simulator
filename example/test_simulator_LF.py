from MF_Sim.LF_Sim import simulator
import numpy as np
import copy
import time


a=simulator.Agent(pos_x=2,pos_y=7,goal_x=20,goal_y=7,direct=0,vel = 1)
b=simulator.Agent(pos_x=4,pos_y=5,goal_x=20,goal_y=5,direct=0,vel = 1)
c=simulator.Agent(pos_x=5,pos_y=1,goal_x=5,goal_y=10,direct=1,vel = 1)
agents=[]
agents.append(a)
agents.append(b)
agents.append(c)

map=np.random.randint(0,2,(30,30))
w = simulator.Full_env(30, 30, False,3,agents,map)


# w =simulator.Full_env(30, 30, True,3)

w.reset()

# w.set_start([[2,7],[4,5],[5,1]])
# w.set_goal([[20,7],[20,5],[5,10]])
# w.set_direct([0,0,1])

start = time.time()
for i in range(100000) :
    time.sleep(0.1)
    w.render()
    if ((i+1)%20)==0:
        w.reset(True)
        #print('w.map[0][0]',w.map[0][0],'w.map[1][1]',w.map[1][1])
    # print('方向1：',w.agents[0].direct)
    # print('原始方向1：',w.agents[0].direct_origin)
    #print('x坐标：',w.agents[0].pos.x)
    #print('y坐标：',w.agents[0].pos.y)
    obs, reward, done,info = w.step([1,2,3])
    print('done',done)
    print(w.agents[0].movable)
    print('reward:',reward)
    print('info',info)
    # print('方向：',w.agents[0].direct)
    # print('原始方向：',w.agents[0].direct_origin)
    # print('x坐标：',w.agents[0].pos.x)
    # print('y坐标：',w.agents[0].pos.y)
end = time.time()
print("sum time:",end-start)
