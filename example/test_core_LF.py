from MF_Sim.LF_Sim import core

a=core.Agent(pos_x=2,pos_y=7,goal_x=20,goal_y=7,direct=0,vel = 1)
b=core.Agent(pos_x=4,pos_y=5,goal_x=20,goal_y=5,direct=0,vel = 1)
c=core.Agent(pos_x=5,pos_y=1,goal_x=5,goal_y=10,direct=1,vel = 1)

print(a)
print(b)
print(c)

agents=[]
agents.append(a)
agents.append(b)
agents.append(c)


