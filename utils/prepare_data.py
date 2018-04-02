#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-2 上午8:40
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com
import os
import numpy as np
from utils.plyfile import PlyData,PlyElement

#create dataset

label_map = {'01':1,'02':2,'03':3}

txtfile = '/home/li/PycharmProjects/classfication/data/datas.text'

def traversal(path, txtfile):
    '''write data paths to a text file with labels(path,label) '''
    with open(txtfile, 'w') as f:
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.sep.join([root, filename])
                dirname = root.split(os.sep)[-1] #os.sep是当前操作系统的路径分隔符
                label = label_map[dirname]
                line = '{},{}\n'.format(filepath, label)
                f.write(line)
                #return filepath, dirname,label

#Point cloud IO
def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


if __name__=='__main__':
    datapath = '/home/li/Desktop/3dmodel/train/'
    traversal(datapath)
    # filenames = os.listdir(datapath)
    # for file in filenames:
    #     label = label_map[int(file)]
    #
    #print(dirname)