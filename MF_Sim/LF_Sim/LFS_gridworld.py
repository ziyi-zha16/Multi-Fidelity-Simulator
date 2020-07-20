#-*- coding: UTF-8 -*-
import numpy as np
import math
import random
import gym
from gym import spaces
import copy
import sys
import pdb
import imageio
from PIL import Image
sys.path.append("..")
from multiagent_particle_envs.multiagent.multi_discrete import MultiDiscrete
from MF_Sim.HF_Sim import random_map, simulator

from gym.utils import seeding

class Agent(object):
    def __init__(self,pos_x=None,pos_y=None,goal_x=None,goal_y=None,direct=-1):
        #position
        self.pos = Coordinate(pos_x,pos_y)      #二维坐标类
        self.pos_origin_x=pos_x     #储存原始信息，便于reset（不开新地图）
        self.pos_origin_y=pos_y
        #目标点
        self.goal = Coordinate(goal_x,goal_y)
        self.goal_origin_x=goal_x    #储存原始信息，便于reset（不开新地图）
        self.goal_origin_y=goal_y
        #action
        self.act = 0        #动作0为停止，1,2,3,4,分别对应右，上，左，下
        #movable
        self.movable = True
        #reach
        self.reach = False
        #朝向
        self.direct = direct            #方向，取0,1,2,3时分别表示右，上，左，下
        self.direct_origin=self.direct
        #crash
        self.crash = False
        #速度
        self.vel = 1.6
        #转向半径
        self.r = 1.71
        #存储上一个state
        self.s_buffer = Coordinate()
        #长宽
        self.L_car = 0.8
        self.W_car = 0.4
        self.color = [255,0,0]
    #设置起点
    def set_start(self, x,y):
        self.pos.x = x
        self.pos_origin_x=x
        self.pos.y = y
        self.pos_origin_y=y

    #设置终点
    def set_goal(self, x,y):
        self.goal.x = x
        self.goal_origin_x=x
        self.goal.y = y
        self.goal_origin_y=y

    #设置方向
    def set_direct(self,x):
        self.direct=x
        self.direct_origin=self.direct

    def reach_set(self):                                    #到达之后重新设置随机终点
        self.goal.x = round(random.randint(1,self.len-2))
        self.goal.y = round(random.randint(1,self.wid-2))
        while((self.goal.x == self.pos.x and self.goal.y == self.pos.y) or ([agent.goal.x, agent.pos.y] in goal_list)):
            self.goal.x = round(random.randint(1,self.len-2))           #若终点和当前位置一致，重新设定
            self.goal.y = round(random.randint(1,self.wid-2))



#坐标类
class Coordinate(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

class world(gym.Env):
    def __init__(self, length=0, width=0,is_random=False, agent_number=1, agents=None, map=None, goal_list=[] ,res=1):
        #地图长宽
        self.len = length
        self.wid = width
        self.res = res #1米每格

        #智能体信息
        self.agent_number=agent_number            #智能体数目
        if agents is not None:
            self.agents = agents
            if agent_number!=len(agents):
                print('warning:agent_number is not equal to len(agents)')
                agent_number=len(agents)            #修正智能体数目
        else:
            self.agents=[Agent() for _ in range(agent_number)]     #使用默认参数构造智能体

        #起点终点
        self.goal_list=goal_list

        self.theta = [0, 0.5, 1, 1.5]       #用于取转向角度0至2pi     

        self.viewer = None
        self.agent_geom_list = None
        self.grid_geom_list = None          #渲染画面用的

        #self.direct_space = 2
        self.cam_range = 25
        self.rewcnt = 0
        self.crashcnt = 0
        self.pcnt = 0
        self.mcnt = 0
        self.rcnt = 0
        self.bonus_map = np.ones((self.len, self.wid))
        self.is_random = is_random

        #构造地图和智能体
        if is_random:                           #地图和智能体均随机
            self.random_fill(length, width)     #随机构建地图和agent的位置和目标
        elif map is None:
            self.map=np.zeros((self.len, self.wid))     #构建全0地图
        else:
            self.map = map
            self.o_map = copy.deepcopy(self.map)            #该地图不包含agents信息，仅包含障碍物
        for agent in self.agents:               #将agent的位置放入地图
            agent.movable = True
            agent.reach = False
            agent.crash = False
            if agent.pos.x!=None and agent.pos.y!=None:
                self.map[agent.pos.x][agent.pos.y] = 1
        

        self.action_space = [spaces.Discrete(5) for _ in range(self.agent_number)]  #定义动作空间

        self.observation_space = []             #设置观测空间
        for agent in self.agents:
            self.observation_space.append(self.observation(agent))

    def observation(self,agent):                    #得到当前的观测
        map = []#四通道图，自己的位置，自己的目标，障碍地图，其他agent的位置
        #自己的位置
        map_self_pos = np.zeros((self.len, self.wid))
        if agent.pos.x!=None and agent.pos.y!=None:
            map_self_pos[agent.pos.x][agent.pos.y] = 1
        map.append(map_self_pos)
        #自己的目标
        map_self_goal = np.zeros((self.len, self.wid))
        if agent.goal.x!=None and agent.goal.y!=None:
            map_self_goal[agent.goal.x][agent.goal.y] = 1
        map.append(map_self_goal)
        #障碍地图
        map.append(self.o_map)
        map_other_pos = np.zeros((self.len, self.wid))
        #其他agent的位置
        for other in self.agents:
            if other is agent: continue
            if agent.pos.x!=None and agent.pos.y!=None:
                map_other_pos[other.pos.x][other.pos.y] = 1
        map.append(map_other_pos)
        #方向
        direct1 = np.zeros((self.len,self.wid))
        direct2 = np.zeros((self.len,self.wid))
        direct1[:][:] = np.cos(np.pi*self.theta[agent.direct])
        direct2[:][:] = np.sin(np.pi*self.theta[agent.direct])
        map.append(direct1)
        map.append(direct2)
    
        return map

    def get_state(self):#[[x,y,direct,vel],[]...]       #获取状态
        state_list = []
        for agent in self.agents:
            state = [agent.pos.x, agent.pos.y, agent.direct, agent.vel]
            state_list.append(state)
        return state_list

    #判空
    def is_empty(self,x,y):                         #若非空，返回1
        return (not bool(self.o_map[x,y]))

    #随机填充
    def random_fill(self, length, width):
        goal_list=[]

        #随机生成地图和位置
        half_wall_width = 1         #一些默认参数
        car_R = 0.5
        door_width = 3
        room_number = 2
        #创建地图
        room_list, wall_list, placeable_area_list = random_map.random_room(map_W = width,
                                                        map_H = length,
                                                        half_wall_width = half_wall_width,
                                                        car_R = car_R,
                                                        door_width = door_width,
                                                        room_number = room_number)
        wall_list = random_map.place_door(room_list,wall_list,half_wall_width,door_width)       #相较于原来的wall_list多了门的信息
        fence_dict = random_map.wall2fence(wall_list)
        
        self.map = np.zeros((length, width))
        self.map[0,:] = 1
        self.map[:,0] = 1
        self.map[length-1,:] = 1
        self.map[:,width-1] = 1
        
        #前四个边墙隔过去
        for k,fence in fence_dict.items():
            if k<4:
                continue
            vertices_x = fence['vertices_x']
            vertices_y = fence['vertices_y']
            for x in range(int(round(vertices_x[0])), int(round(vertices_x[2]))):
                for y in range(int(round(vertices_y[0])), int(round(vertices_y[2]))):
                    self.map[x,y] = 1
        
        self.o_map = copy.deepcopy(self.map)
        
        agent_dict = simulator.random_agent(placeable_area_list, self.agent_number)
        # np.save('agent_dict2.npy',agent_dict)
        # #np.save('agent_dict.npy',agent_dict)
        agent_list = agent_dict['main_group']
        
        for i in range(len(self.agents)):
        #for agent in self.agents:
            agent = self.agents[i]
            agent_HFS = agent_list[i]
            agent.act = 0
            agent.movable = True
            agent.reach = False
            agent.crash = False
            #agent.vel = 1.47
            
            agent.direct = math.floor(agent_HFS['init_theta']*4/6.28)   #确定随机方向
            assert agent.direct>=0
            assert agent.direct<4
            agent.direct_origin=agent.direct

            agent.pos.x = int(round(agent_HFS['init_x']))
            agent.pos.y = int(round(agent_HFS['init_y']))
            while self.map[agent.pos.x,agent.pos.y] == 1:
                agent.pos.x = round(random.randint(1,self.len-2))
                agent.pos.y = round(random.randint(1,self.wid-2))
            self.map[agent.pos.x][agent.pos.y] = 1
            agent.pos_origin_x=agent.pos.x          #记录原始信息
            agent.pos_origin_y=agent.pos.y

            agent.goal.x = int(round(agent_HFS['init_target_x']))
            agent.goal.y = int(round(agent_HFS['init_target_y']))
            while self.o_map[agent.goal.x,agent.goal.y] == 1:
                agent.goal.x = round(random.randint(1,self.len-2))
                agent.goal.y = round(random.randint(1,self.wid-2))
            goal_list.append(Coordinate(agent.goal.x,agent.goal.y))
            agent.goal_origin_x=agent.goal.x
            agent.goal_origin_y=agent.goal.y
      
        self.goal_list = copy.deepcopy(goal_list)


    #设置起点 传入一个元素形式为C=[x,y]的list[C0,C1......]
    def set_start(self, start):
        cnt = 0
        for c in start:
            if c[0] >= self.len or c[1] >= self.wid or c[0]<0 or c[1]<0 :
                print("Wrong setting for start(",c[0],",",c[1],")")
                return(False)
            if self.agents[cnt].pos.x!=None and self.agents[cnt].pos.y!=None:
                self.map[self.agents[cnt].pos.x][self.agents[cnt].pos.y] = 0     #清除地图上原来的坐标
            self.map[c[0], c[1]] = 1
            self.agents[cnt].set_start(c[0],c[1])
            cnt = cnt + 1
            
    #设置终点，格式同起点
    def set_goal(self, goal):
        self.goal_list = goal
        for (a,g) in zip(self.agents, self.goal_list):
            a.set_goal(g[0],g[1])
    

    def step(self, action):
        #按照action给每个agent更新位置
        obs = []
        #direct = []
        reward = []
        done = []
        info = []
        for (a, agent) in zip(action,self.agents):#action操作0-5,0不动，1右转，2前进，3左转，4后退
            agent.act = a
            #print(a)
            if agent.movable:
                #直行
                if agent.act == 2:
                    x_new = agent.pos.x + agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    y_new = agent.pos.y + agent.vel*np.sin(np.pi*self.theta[agent.direct])
                    direct_new = agent.direct
                #后退
                elif agent.act == 4:
                    x_new = agent.pos.x - agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    y_new = agent.pos.y - agent.vel*np.sin(np.pi*self.theta[agent.direct])
                    direct_new = (agent.direct+2)%4
                #左行
                elif agent.act == 3:
                    x_new = agent.pos.x - agent.vel*np.sin(np.pi*self.theta[agent.direct])  #更新坐标
                    y_new = agent.pos.y + agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    direct_new = round(agent.direct+1)%4                               #新的方向
                #右行
                elif agent.act == 1:
                    x_new = agent.pos.x + agent.vel*np.sin(np.pi*self.theta[agent.direct])  #更新坐标
                    y_new = agent.pos.y - agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    direct_new = round(agent.direct-1)%4                               #新的方向
                #不动
                else:
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                    direct_new = agent.direct
                
                x_new = int(round(x_new))                   #对坐标取整
                y_new = int(round(y_new)) 

                if x_new > self.len-1 or x_new<0 or y_new>self.wid-1 or y_new<0:    #如果碰撞（地图越界）
                    agent.crash = True
                    #print("map_crash x_new:",x_new, "y_new:", y_new)   #如果碰撞，坐标变回当前坐标
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                    direct_new = agent.direct
                elif self.o_map[x_new][y_new] == True:              #如果有障碍物 
                    agent.crash = True
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                    direct_new = agent.direct
                
                for other in self.agents:       #判断是否与其他agent相撞
                    if other == agent:
                        continue
                    if other.pos.x == x_new and other.pos.y == y_new:
                        other.crash = True
                        agent.crash = True
                        #print("car_crash")
                        x_new = agent.pos.x
                        y_new = agent.pos.y
                        direct_new = agent.direct

                self.map[agent.pos.x][agent.pos.y] = 0      #位置更新
                agent.s_buffer.x = agent.pos.x              #将当前坐标储存为前一个坐标
                agent.s_buffer.y = agent.pos.y
                agent.pos.x = x_new                         #更新当前坐标和方向以及agent的位置
                agent.pos.y = y_new
                agent.direct = direct_new

                self.map[agent.pos.x][agent.pos.y] = 1
                self.bonus_map[agent.pos.x][agent.pos.y]+=1 #表示agent到达地图上的该位置的次数
            if agent.pos.x <= agent.goal.x + 1 and agent.pos.x >= agent.goal.x - 1 and agent.pos.y <= agent.goal.y + 1 and agent.pos.y>= agent.goal.y - 1:
                agent.reach = True      #当agent到达目标点及其周围的8个格点时，视为到达
                #agent.reach_set()
                agent.movable = False
        for agent in self.agents:       #返回各种信息
            obs.append(self.observation(agent))
            info.append([agent.pos.x,agent.pos.y,agent.crash,agent.reach])#经过reward之后，reach和crash的信息会被处理掉，所以要放到reward之前
            done.append(agent.reach)
            reward.append(self.reward(agent))
        self.observation_space=np.array(obs)

        return np.array(obs), np.array(reward), done, info


    def reset(self,new_map=None):#怎么随机重置？
        if new_map is not None:                         #使用新地图
            #print('new map')
            self.random_fill(self.len,self.wid)
            #self.viewer = None                         #加上这一句会开新的窗口
            self.agent_geom_list = None                 #为了使地图能在可视化界面被重新构造
            self.grid_geom_list = None
        else:
            #print('old map')
            self.map=copy.deepcopy(self.o_map)          #通过障碍物地图来还原地图
            for agent in self.agents:                   #非随机情况下，还原原始信息，重新进行训练
                agent.set_start(agent.pos_origin_x,agent.pos_origin_y)  #还原agent的起始坐标和最终目标
                agent.set_goal(agent.goal_origin_x,agent.goal_origin_y)
                agent.act = 0
                agent.movable = True
                agent.reach = False
                agent.direct = agent.direct_origin      #还原agent的起始方向
                self.crash = False
                self.s_buffer = Coordinate()
                if agent.pos.x!=None and agent.pos.y!=None:         #将agent的位置信息填入地图中
                    self.map[agent.pos.x][agent.pos.y] = 1
        
        obs = []
        for agent in self.agents:
            obs.append(self.observation(agent))

        self.observation_space=np.array(obs)        #返回观测信息
        return np.array(obs)


    def reward(self,agent):#fai(x) - fai'(x)

        def dist(self, c1, c2):
            return math.sqrt((c1.x-c2.x)*(c1.x-c2.x) + (c1.y-c2.y)*(c1.y-c2.y))
        rew = 0
        #print(agent.pos.x, agent.pos.y, agent.s_buffer.x, agent.s_buffer.y)
        #self.rewcnt += 1
        #dists = math.sqrt((agent.pos.x-agent.goal.x)*(agent.pos.x-agent.goal.x)+(agent.pos.y-agent.goal.y)*(agent.pos.y-agent.goal.y))
        
        #if dist(self, agent.pos, agent.goal)>2:
        #    rew = 0.7*(-(dist(self, agent.pos, agent.goal) - dist(self, agent.s_buffer, agent.goal)))-0.1
        #else:
        #    rew = -0.1
        rew = 0.7*(-(dist(self, agent.pos, agent.goal) - dist(self, agent.s_buffer, agent.goal)))-0.1 #定义奖励，-0.1为每一步的固定奖励
        #rho = 20 #bonus parameter
        #rew+=rho*(1/self.bonus_map[agent.pos.x][agent.pos.y])
        if agent.crash:
            rew -= 5
            agent.crash = 0
            #print("erase crash")
        if agent.reach:
            rew += 5
            agent.reach = False
            #self.rcnt += 1
            #print("agent reach", self.rcnt)
        #if rew<-4:
        #    self.crashcnt += 1
        #    #print("crash_reward:", rew, "rewcnt:", self.rewcnt)
        #elif rew<0:
        #    self.mcnt += 1
        #else:
        #    self.pcnt +=1
        #
        #if self.rcnt%1000==0:
        #    print("agent reach",self.rcnt)
        #    print("crash:",self.crashcnt, "-reward:",self.mcnt,"reach:",self.rcnt, "+reward:",self.pcnt)
        #    self.crashcnt = 0
        #    self.mcnt = 0
        #    self.pcnt = 0
        #    self.rcnt = 0

        #print(rew)
        return 0.2*rew
                
    def render(self, mode = 'human'):
        if self.viewer is None:
            from multiagent_particle_envs.multiagent import rendering
            self.viewer = rendering.Viewer(800,800)         #设置窗口大小

        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent_particle_envs.multiagent import rendering
            self.viewer.set_bounds(0, 0+2*self.cam_range, 0, 0+2*self.cam_range)        #设置能看到的范围
            self.agent_geom_list = []
            self.grid_geom_list = []
            for agent in self.agents:
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform

                half_l = agent.L_car/2.0
                half_w = agent.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*agent.color,alpha = 0.4)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                #geom = rendering.make_line((0,0),(half_l,0))
                #geom.set_color(1.0,0.0,0.0,alpha = 1)
                #xform = rendering.Transform()
                #geom.add_attr(xform)
                #geom.add_attr(total_xform)
                #agent_geom['front_line']=(geom,xform)
                self.agent_geom_list.append(agent_geom)
            for x in range(self.wid):
                for y in range(self.len):
                    grid_geom={}
                    total_xform = rendering.Transform()
                    grid_geom['total_xform'] = total_xform

                    geom = rendering.make_polygon([[x+0.5,y+0.5],[x-0.5, y+0.5], [x-0.5, y-0.5], [x+0.5, y-0.5]])
                    if self.map[x][y] == 1:
                        geom.set_color(255,255,255, alpha = 0.4)
                    elif self.map[x][y] == 0:
                        geom.set_color(0,255,0,alpha=0.5)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    grid_geom['grid']=(geom,xform)
                    self.grid_geom_list.append(grid_geom)

            self.viewer.geoms = []
            for agent_geom in self.agent_geom_list:
                #self.viewer.add_geom(agent_geom['target_circle'][0])
                #for geom in agent_geom['laser_line']:
                #    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
            for grid_geom in self.grid_geom_list:
                self.viewer.add_geom(grid_geom['grid'][0])
                #self.viewer.add_geom(agent_geom['front_line'][0])
                #self.viewer.add_geom(agent_geom['back_line'][0])
        for agent,agent_geom in zip(self.agents,self.agent_geom_list):
            
            #for idx,laser_line in enumerate(agent_geom['laser_line']):
            #        laser_line[1].set_scale(agent.laser_state[idx],agent.laser_state[idx]) 
            #agent_geom['front_line'][1].set_rotation(agent.state.phi)
            #agent_geom['target_circle'][1].set_translation(agent.state.target_x,agent.state.target_y)
            agent_geom['total_xform'].set_rotation(np.pi*self.theta[agent.direct])
            agent_geom['total_xform'].set_translation(agent.pos.x,agent.pos.y)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
            