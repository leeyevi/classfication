#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-2 上午8:40
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com
import os
import numpy as np
from utils.plyfile import PlyData,PlyElement
# Draw point cloud
from utils.eulerangles import euler2mat

#create dataset

label_map = {'01':1,'02':2,'03':3}

txtfile = '/home/li/PycharmProjects/classfication/data/datas.text'

def getDataFiles(list_filename):
    '''read txt return a array(filename,label)'''
    full = [line.rstrip() for line in open(list_filename)]
    for i in range(len(full)):
        full[i] = full[i].split(',')
    return full


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



if __name__=='__main__':
    #datapath = '/home/li/Desktop/3dmodel/train/'
    #traversal(datapath)
    # filenames = os.listdir(datapath)
    # for file in filenames:
    #     label = label_map[int(file)]
    #
    #print(dirname)
    datas = getDataFiles('/home/li/PycharmProjects/classfication/data/datas.text')
    #narr = np.zeros_like(filenames)
    
    # for file in filenames:
    #     file = file.split(',')
        #data = narr.append(file)



    #data = filenames[:,:]
    #label = filenames[:,1]
    print(data for data in datas)