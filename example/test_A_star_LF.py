# -*- coding: utf-8 -*-
import os,sys       #这几行（2~4）在安装了之后可以省略
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 

from MF_Sim.LF_Sim import simulator

import numpy as np
import time
 
class Node_Elem:
     """
     开放列表和关闭列表的元素类型，parent用来在成功的时候回溯路径
     """
     def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist
         
class A_Star:
     """
     A星算法实现类
     """
     #注意w,h两个参数，如果你修改了地图，需要传入一个正确值或者修改这里的默认参数
     #s代表起点，e代表终点
     def __init__(self, s_x, s_y, e_x, e_y, L=30, W=30,test_map=[]):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y
        
        self.length = L
        self.width = W
        self.test_map=test_map

        self.open = []
        self.close = []
        self.path = []

     #查找路径的入口函数
     def find_path(self):
        #构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        while True:
            #扩展F值最小的节点
            self.extend_round(p)
            #如果开放列表为空，则不存在路径，返回
            if not self.open:
                return
             #获取F值最小的节点
            idx, p = self.get_best()
             #找到路径，生成路径，返回
            if self.is_target(p):
                self.make_path(p)
                return
             #把此节点压入关闭列表，并从开放列表里删除
            self.close.append(p)
            del self.open[idx]
             
     def make_path(self,p):
         #从结束点回溯到开始点，开始点的parent == None
        while p:
            self.path.insert(0,(p.x, p.y))
            p = p.parent
         
     def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y
         
     def get_best(self):
        best = None
        bv = 1000000 #如果你修改的地图很大，可能需要修改这个值
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)#获取F值
            if value < bv:#比以前的更好，即F值更小
                best = i
                bv = value
                bi = idx
        return bi, best
        
     def get_dist(self, i):
        # F = G + H
        # G 为已经走过的路径长度， H为估计还要走多远
        # 这个公式就是A*算法的精华了。
        return i.dist + ((self.e_x-i.x)+(self.e_y-i.y))
        
     def extend_round(self, p):
        #只能走上下左右四个方向
        xs = (0, -1, 1, 0)
        ys = (-1, 0, 0, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
             #无效或者不可行走区域，则忽略
            if not self.is_valid_coord(new_x, new_y):
                continue
            #构造新的节点
            node = Node_Elem(p, new_x, new_y, p.dist+1)
            #新节点在关闭列表，则忽略
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                #新节点在开放列表
                if self.open[i].dist > node.dist:
                    #现在的路径到比以前到这个节点的路径更好
                    #则使用现在的路径
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.open.append(node)
         
     def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False
         
     def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1
         
     def is_valid_coord(self, x, y):
        if x < 0 or x >= self.length or y < 0 or y >= self.width:
            return False
        return self.test_map[x][y] != 1
      
def find_path(s_x,s_y,e_x,e_y,o_map):
    a_star = A_Star(s_x, s_y, e_x, e_y, 30 ,30,o_map)
    a_star.find_path()
    path = a_star.path
    print ('路径长度为',len(path))
    return path

def act_judge(pos1,pos2,direct):    #用于判断方向为direct时，agent经过什么动作能从pos1变换到pos2
    if pos1 is None or pos2 is None:
        return 0
    action=None
    for act_test in range(5):   #遍历所有动作，依此判断经过哪个动作能够从pos1变换到pos2
        #直行
        if act_test == 2:
            x_new = pos1[0] + np.cos(np.pi*theta[direct])
            y_new = pos1[1] + np.sin(np.pi*theta[direct])
        #后退
        elif act_test == 4:
            x_new = pos1[0] - np.cos(np.pi*theta[direct])
            y_new = pos1[1] - np.sin(np.pi*theta[direct])
        #左行
        elif act_test == 3:
            x_new = pos1[0] - np.sin(np.pi*theta[direct])
            y_new = pos1[1] + np.cos(np.pi*theta[direct])
        #右行
        elif act_test == 1:
            x_new = pos1[0] + np.sin(np.pi*theta[direct])
            y_new = pos1[1] - np.cos(np.pi*theta[direct])
        #不动
        else:
            x_new = pos1[0]
            y_new = pos1[1]
        if x_new==pos2[0] and y_new==pos2[1]:
            action=act_test
            return action

if __name__ == "__main__":
    theta=[0, 0.5, 1, 1.5]          #角度辅助参数
    agent=simulator.Agent(vel=1)    #将agent的速度调整为1
    agents=[]
    agents.append(agent)
    env=simulator.Full_env(30, 30, True,1,agents)
    o_map=env.get_o_map()
    state=env.get_state()
    pos_x=state[0][0]
    pos_y=state[0][1]
    goal_x=state[0][2]
    goal_y=state[0][3]
    direct=state[0][4]
    vel=state[0][5]
    print('agent的初始状态：pos.x=',pos_x,'pos.y=',pos_y,
    'goal.x=',goal_x,'goal.y=',goal_y,'direct=',direct,'vel=',vel)

    path=find_path(pos_x,pos_y,goal_x,goal_y,o_map)
    print('路径：',path)

    pos_former=None     #用于储存前一个点的坐标
    for pos in path:
        time.sleep(0.1)
        state=env.get_state()
        direct=state[0][4]         #得到朝向
        act=act_judge(pos_former,pos,direct)
        if act>0 and act<=4:
            env.render(0)
            env.step([act])
        pos_former=pos
