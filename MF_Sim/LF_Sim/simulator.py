#-*- coding: UTF-8 -*-
import numpy as np
import math
import random
import gym
from gym import spaces
import copy

from gym.utils import seeding
import sys
# import os                   #这几行（11~13）在安装了之后可以省略
# parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
# sys.path.insert(0,parentdir) 

from MF_Sim.LF_Sim.core import *
from MF_Sim.HF_Sim import random_map,simulator

class Full_env(gym.Env):
    def __init__(self, length=0, width=0,is_random=False, agent_number=1, agents=None, o_map=None, goal_list=[] ,res=1):
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
        self.local_viewer=None
        self.agent_geom_list = None
        self.local_agent_geom_list = None
        self.grid_geom_list = None          #渲染画面用的
        self.local_grid_geom_list = None

        self.cam_range = round(self.len/2)+5
        # self.rewcnt = 0
        # self.crashcnt = 0
        # self.pcnt = 0
        # self.mcnt = 0
        # self.rcnt = 0
        self.bonus_map = np.ones((self.len, self.wid))
        self.is_random = is_random

        self.seed()

        #构造地图和智能体
        if is_random:                           #地图和智能体均随机
            self.random_fill(length, width)     #随机构建地图和agent的位置和目标
        elif o_map is None:
            self.o_map=np.zeros((self.len, self.wid))     #构建全0地图
            self.map = copy.deepcopy(self.o_map)
        else:
            self.o_map = o_map
            self.map = copy.deepcopy(self.o_map)            #该地图不包含agents信息，仅包含障碍物
        for agent in self.agents:               #将agent的位置放入地图，并定义好本地地图的大小
            agent.set_localmap_size(3*self.len,3*self.wid)
            if agent.pos.x!=None and agent.pos.y!=None:
                self.map[agent.pos.x][agent.pos.y] = 1
        

        self.action_space = [spaces.Discrete(5) for _ in range(self.agent_number)]  #定义动作空间

        self.observation_space = []             #设置观测空间
        for agent in self.agents:
            self.observation_space.append(self.observation(agent))
        self.observation_space=np.array(self.observation_space)


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
        direct = np.zeros((self.len,self.wid))
        direct[:][:] = agent.direct
        #direct=agent.direct
        map.append(direct)
        #得到两个地图的坐标偏差
        deviation_x= np.zeros((self.len,self.wid))
        deviation_y= np.zeros((self.len,self.wid))
        deviation_x[:][:]=agent.deviation_x
        deviation_y[:][:]=agent.deviation_y
        map.append(deviation_x)
        map.append(deviation_y)

        return map

    def get_o_map(self):       #得到障碍物地图
        return self.o_map

    def get_state(self):#[[x,y,direct,vel],[]...]       #获取状态
        state_list = []
        for agent in self.agents:
            state = [agent.pos.x, agent.pos.y, agent.goal.x, agent.goal.y,agent.direct, agent.vel]
            state_list.append(state)
        return state_list

    #判空
    def is_empty(self,x,y):                         #若非空，返回1
        return (not bool(self.o_map[x,y]))

    #随机填充
    def random_fill(self, length, width,half_wall_width = 1,car_R = 0.5,door_width = 3,room_number = 2):
        goal_list=[]
        #随机生成地图和位置
        #创建地图
        room_list, wall_list, placeable_area_list = random_map.random_room(map_W = width,
                                                        map_H = length,
                                                        half_wall_width = half_wall_width,
                                                        car_R = car_R,
                                                        door_width = door_width,
                                                        room_number = room_number,
                                                        np_random=self.np_random)
        wall_list = random_map.place_door(room_list,wall_list,half_wall_width,door_width,self.np_random)       #相较于原来的wall_list多了门的信息
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
                    self.map[length-y-1,x] = 1
        
        self.o_map = copy.deepcopy(self.map)
        
        agent_dict = simulator.random_agent(placeable_area_list, self.agent_number,self.np_random)
        # np.save('agent_dict2.npy',agent_dict)
        # #np.save('agent_dict.npy',agent_dict)
        agent_list = agent_dict['main_group']
        
        for i in range(len(self.agents)):
        #for agent in self.agents:
            agent = self.agents[i]
            agent_HFS = agent_list[i]
            agent.reset() 
            #agent.vel = 1.47
            
            agent.direct = math.floor(agent_HFS['init_theta']*4/6.28)   #确定随机方向
            assert agent.direct>=0
            assert agent.direct<4
            agent.direct_origin=agent.direct

            agent.pos.x = length-int(round(agent_HFS['init_y']))-1
            agent.pos.y = int(round(agent_HFS['init_x']))
            while self.map[agent.pos.x,agent.pos.y] == 1:
                agent.pos.x = round(random.randint(1,length-2))
                agent.pos.y = round(random.randint(1,width-2))
            self.map[agent.pos.x][agent.pos.y] = 1
            agent.pos_origin_x=agent.pos.x          #记录原始信息
            agent.pos_origin_y=agent.pos.y

            agent.goal.x = length-int(round(agent_HFS['init_target_y']))-1
            agent.goal.y = int(round(agent_HFS['init_target_x']))
            while self.o_map[agent.goal.x,agent.goal.y] == 1:
                agent.goal.x = round(random.randint(1,length-2))
                agent.goal.y = round(random.randint(1,width-2))
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

    #设置各个智能体的方向
    def set_direct(self,direct):
        cnt = 0
        for c in direct:
            if c>3 or c<0:
                print("Wrong setting for direct(",c,")")
                return(False)
            self.agents[cnt].set_direct(c)
            cnt = cnt + 1


    def step(self, action):
        #按照action给每个agent更新位置
        obs = []
        #direct = []
        reward = []
        done = []
        info = []
        #deviation_list=[]
        for (a, agent) in zip(action,self.agents):#action操作0-5,0不动，1右转，2前进，3左转，4后退
            agent.act = a
            #print(a)
            deviation=agent.get_deviation()
            # print(deviation)
            # deviation_list.append(deviation)
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
                agent.localmap_update(self.map)      #根据agent的位置好更新localmap

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


    def reset(self,new_map=False):#怎么随机重置？
        if new_map is not False:                         #使用新地图
            #print('new map')
            self.random_fill(self.len,self.wid)
            #self.viewer = None                         #加上这一句会开新的窗口
            self.agent_geom_list = None                 #为了使地图能在可视化界面被重新构造
            self.grid_geom_list = None
        else:
            #print('old map')
            self.map=copy.deepcopy(self.o_map)          #通过障碍物地图来还原地图
            for agent in self.agents:                   #非随机情况下，还原原始信息，重新进行训练
                agent.reset()
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
            agent.crash = False
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
                
    def render(self,num=None, mode = 'human'):
        if self.viewer is None:
            from MF_Sim.HF_Sim import rendering
            self.viewer = rendering.Viewer(800,800)         #设置窗口大小

        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from MF_Sim.HF_Sim import rendering
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
            for x in range(self.len):
                for y in range(self.wid):
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
        
        self.viewer.render(return_rgb_array = mode=='rgb_array')
        if num is not None:
            if num>=0 and num<self.agent_number:
                self.local_render(num)
            else:
                print('warming:agent index in render() is out of range,agent_number=',self.agent_number)
                sys.exit()

    def local_render(self, num,mode = 'human'):
        if self.local_viewer is None:
            from MF_Sim.HF_Sim import rendering
            self.local_viewer = rendering.Viewer(800,800)         #设置窗口大小

        if self.local_agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from MF_Sim.HF_Sim import rendering
            self.local_viewer.set_bounds(0, 0+3*self.len+20, 0, 0+3*self.wid+20)        #设置能看到的范围
            self.local_agent_geom_list = []
            self.local_grid_geom_list = []
            agent=self.agents[num]
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

            self.local_agent_geom_list.append(agent_geom)
            for x in range(agent.localmap_L):
                for y in range(agent.localmap_W):
                    grid_geom={}
                    total_xform = rendering.Transform()
                    grid_geom['total_xform'] = total_xform

                    geom = rendering.make_polygon([[x+0.5,y+0.5],[x-0.5, y+0.5], [x-0.5, y-0.5], [x+0.5, y-0.5]])
                    if agent.localmap[x][y] == 1:
                        geom.set_color(255,255,255, alpha = 0.4)
                    elif agent.localmap[x][y] == 0:
                        geom.set_color(0,255,0,alpha=0.5)
                    elif agent.localmap[x][y] == -1:
                        geom.set_color(0,0,0,alpha=0.5)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    grid_geom['grid']=(geom,xform)
                    self.local_grid_geom_list.append(grid_geom)

            self.local_viewer.geoms = []
            for agent_geom in self.local_agent_geom_list:
                self.local_viewer.add_geom(agent_geom['car'][0])
            for grid_geom in self.local_grid_geom_list:
                self.local_viewer.add_geom(grid_geom['grid'][0])

        for agent_geom in self.local_agent_geom_list:
            agent_geom['total_xform'].set_rotation(np.pi*self.theta[agent.direct])
            agent_geom['total_xform'].set_translation(agent.localpos.x,agent.localpos.y)

        self.local_agent_geom_list=None
        self.local_grid_geom_list = None
        self.local_viewer.render(return_rgb_array = mode=='rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]