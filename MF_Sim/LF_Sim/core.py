#-*- coding: UTF-8 -*-
import numpy as np
import math
import random
import gym
from gym import spaces

class Agent(object):
    def __init__(self,pos_x=None,pos_y=None,goal_x=None,goal_y=None,direct=-1,vel = 1.6,L_car = 0.8,W_car = 0.4,
                camera_L=5,camera_W=5,localmap_L=100,localmap_W=100,color = [255,0,0]):
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
        self.vel = vel
        #存储上一个state
        self.s_buffer = Coordinate()
        #长宽
        self.L_car = L_car
        self.W_car = W_car
        self.color = color
        #agent上搭载的摄像头能看到的范围以及通过摄像头得到的地图
        self.camera_L=camera_L
        self.camera_W=camera_W
        #agent自带的本机地图的设置
        self.localmap_L=localmap_L          
        self.localmap_W=localmap_W
        self.localmap=np.zeros((localmap_L,localmap_W))
        self.localmap=self.localmap-1           #-1代表未知区域，初始状态下所有区域均为未知
        self.localpos=Coordinate(round(localmap_L/2),round(localmap_W/2))    #agent在localmap中的初始坐标
        self.deviation_x=None                      #agent在localmap中的坐标和map中坐标的偏差值
        self.deviation_y=None

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

    #设置摄像头能看到的范围
    def set_camera_range(self,camera_L,camera_W):
        self.camera_L=camera_L
        self.camera_W=camera_W

    #得到localmap和map的偏差值
    def get_deviation(self):
        deviation=[]
        self.deviation_x=self.localpos.x-self.pos.x                      #agent在localmap中的坐标和map中坐标的偏差值
        self.deviation_y=self.localpos.y-self.pos.y
        deviation.append(self.deviation_x)
        deviation.append(self.deviation_y)
        return deviation

    #改变localmap的大小
    def set_localmap_size(self,L,W):
        self.localmap_L=L          
        self.localmap_W=W
        self.localmap=np.zeros((L,W))
        self.localmap=self.localmap-1
        self.localpos=Coordinate(round(L/2),round(W/2))                         #agent在localmap中的初始坐标

    #更新localmap
    def localmap_update(self,map):
        x_max,y_max=map.shape                               #得到x,y所能取到的最大值
        self.localpos.x=self.deviation_x+self.pos.x                      #更新agent在localmap中的坐标
        self.localpos.y=self.deviation_y+self.pos.y
        mark=False                                      #遮挡记号，用于判断摄像头两侧视角是否有阻拦
        if self.direct==0:                              #得到照相机探照范围的索引,并根据map中的障碍物信息更新localmap
            x1=self.pos.x+1
            x2=self.pos.x+1+self.camera_L
            y1=self.pos.y-round((self.camera_W-1)/2)-1
            y1=max(-1,y1)                               #为了range(self.pos.y,y1,-1)能够取到0
            y2=self.pos.y+round((self.camera_W-1)/2)+1
            y2=min(y2,y_max-1)
            for y in range(self.pos.y,y1,-1):
                for x in range(x1,x2):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1     #如果被障碍物遮挡，障碍物后边的部分仍然维持为-1(unknow)
                        if y!=self.pos.y and x==x1 and map[self.pos.x][y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break
            for y in range(self.pos.y,y2):
                for x in range(x1,x2):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1     #如果被障碍物遮挡，障碍物后边的部分仍然维持为-1(unknow)
                        if y!=self.pos.y and x==x1 and map[self.pos.x][y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break

        elif self.direct==1:
            x1=self.pos.x-round((self.camera_W-1)/2)-1
            x1=max(-1,x1)
            x2=self.pos.x+round((self.camera_W-1)/2)+1
            x2=min(x2,x_max-1)
            y1=self.pos.y+1
            y2=self.pos.y+1+self.camera_L
            for x in range(self.pos.x,x1,-1):
                for y in range(y1,y2):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1
                        if x!=self.pos.x and y==y1 and map[x][self.pos.y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break
            for x in range(self.pos.x,x2):
                for y in range(y1,y2):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1
                        if x!=self.pos.x and y==y1 and map[x][self.pos.y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break   

        elif self.direct==2:
            x1=self.pos.x-1
            x2=self.pos.x-1-self.camera_L
            y1=self.pos.y-round((self.camera_W-1)/2)-1
            y1=max(-1,y1)
            y2=self.pos.y+round((self.camera_W-1)/2)+1
            y2=min(y2,y_max-1)
            for y in range(self.pos.y,y1,-1):
                for x in range(x1,x2,-1):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1     #如果被障碍物遮挡，障碍物后边的部分仍然维持为-1(unknow)
                        if y!=self.pos.y and x==x1 and map[self.pos.x][y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break
            for y in range(self.pos.y,y2):
                for x in range(x1,x2,-1):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1     #如果被障碍物遮挡，障碍物后边的部分仍然维持为-1(unknow)
                        if y!=self.pos.y and x==x1 and map[self.pos.x][y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break

        elif self.direct==3:
            x1=self.pos.x-round((self.camera_W-1)/2)-1
            x1=max(-1,x1)
            x2=self.pos.x+round((self.camera_W-1)/2)+1
            x2=min(x2,x_max-1)
            y1=self.pos.y-1
            y2=self.pos.y-1-self.camera_L
            for x in range(self.pos.x,x1,-1):
                for y in range(y1,y2,-1):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1
                        if x!=self.pos.x and y==y1 and map[x][self.pos.y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break
            for x in range(self.pos.x,x2):
                for y in range(y1,y2,-1):
                    if map[x][y]==0:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=0
                    else:
                        self.localmap[x+self.deviation_x][y+self.deviation_y]=1
                        if x!=self.pos.x and y==y1 and map[x][self.pos.y]==1:
                            mark=True
                        break
                if mark==True:
                    mark=False
                    break   
    
    #得到localmap的函数接口
    def get_localmap(self):
        return self.localmap

        
    def reset(self):
        self.set_start(self.pos_origin_x,self.pos_origin_y)  #还原agent的起始坐标和最终目标
        self.set_goal(self.goal_origin_x,self.goal_origin_y)
        self.act = 0
        self.movable = True
        self.reach = False
        self.crach = False
        self.direct = self.direct_origin      #还原agent的起始方向
        self.localmap=np.zeros((self.localmap_L,self.localmap_W))
        self.localmap=self.localmap-1           #-1代表未知区域，初始状态下所有区域均为未知
        self.localpos=Coordinate(round(self.localmap_L/2),round(self.localmap_W/2))    #agent在localmap中的初始坐标
        self.deviation_x=None                      #agent在localmap中的坐标和map中坐标的偏差值
        self.deviation_y=None

    def reach_set(self,len,wid,map):                                    #到达之后重新设置随机终点
        self.movable = True
        self.goal.x = round(random.randint(1,len-2))
        self.goal.y = round(random.randint(1,wid-2))
        while((self.goal.x == self.pos.x and self.goal.y == self.pos.y) or (map[self.goal.x][self.goal.y]==1)):
            self.goal.x = round(random.randint(1,len-2))           #若终点和当前位置一致，重新设定
            self.goal.y = round(random.randint(1,wid-2))

#坐标类
class Coordinate(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y