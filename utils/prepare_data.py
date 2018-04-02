#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-2 上午8:40
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com
import os
import numpy as np

label_map = {'01':1,'02':2,'03':3}

def traversal(path):
    with open('/home/li/PycharmProjects/classfication/data/datas.text', 'w') as f:
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.sep.join([root, filename])
                dirname = root.split(os.sep)[-1] #os.sep是当前操作系统的路径分隔符
                label = label_map[dirname]
                line = '{},{}\n'.format(filepath, label)
                f.write(line)
                #return filepath, dirname,label

if __name__=='__main__':
    datapath = '/home/li/Desktop/3dmodel/train/'
    traversal(datapath)
    # filenames = os.listdir(datapath)
    # for file in filenames:
    #     label = label_map[int(file)]
    #
    #print(dirname)