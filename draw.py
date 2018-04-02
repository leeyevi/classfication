#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-22 上午9:56
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com

import numpy as np
import random
import matplotlib.pyplot as plt
from utils.plyfile import PlyData,PlyElement
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def load_ply_normal(filename, point_num):
#     plydata = PlyData.read(filename)
#     pc = plydata['normal'].data[:point_num]
#     pc_array = np.array([[x, y, z] for x,y,z in pc])
#     return pc_array

def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def rdblock(arr, n, alpha):
    l = arr.shape[0]
    m = int(n*alpha)
    start = random.randint(0,l-m)
    length = random.randint(0, m)
    if start == 0:
        A = arr[length+1:l,:]
    else:
        part1 = arr[:start,:]
        part2 = arr[start+length:,:]
        A = np.concatenate((part1,part2), axis=0)
    return A

def rdmv(arr, point_num):
    l = arr.shape[0]
    num = l -point_num
    for i in range(num):
        seed = random.randint(0, arr.shape[0]-1)
        arr = np.delete(arr, seed, axis=0)
    return arr


def get_imcomplete(filename, point_num):

    plydata = PlyData.read(filename)
    plength = plydata['vertex'].data.size
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    seed = random.randint(0,10)
    for i in range(seed):
        s = random.randint(0,100)
        pc_array = rdblock(pc_array, s, 1)
    #len = pc_array.shape(0)
    #np.delete(a, 1, axis=0)
    #num = (plength - point_num)

    pc_array = rdmv(pc_array,point_num)

    return pc_array

def vis3D(x,y,z,):
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    #plt.show()
    plt.savefig(str(i)+'test2png.png', dpi=100)


if __name__=='__main__':
    datapath = '/home/li/Desktop/3dmodel/p01/0125.ply'
    #pc = load_ply_data(datapath, 1024)
    pc = get_imcomplete(datapath, 1024)
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    vis3D(x,y,z)
    #print(pc.shape[0])
