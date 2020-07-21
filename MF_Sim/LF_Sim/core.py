#-*- coding: UTF-8 -*-
import numpy as np
import math
import random
import gym
from gym import spaces

class Agent(object):
    def __init__(self,pos_x=None,pos_y=None,goal_x=None,goal_y=None,direct=-1,
                    vel = 1.6,L_car = 0.8,W_car = 0.4,color = [255,0,0]):
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