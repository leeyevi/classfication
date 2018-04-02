#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-2 下午3:48
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com
import os


'''1.路径获取'''
basedir = os.path.dirname(os.path.abspath(__file__))#返回当前文件目录
print(basedir)
print(os.path.dirname(__file__))#返回当前文件目录
print(os.path.dirname(os.path.dirname(__file__)))#获取当前文件所在目录的上级目录
print(os.path.abspath(__file__))#获取当前文件的绝对路径

if not os.path.exists(basedir):
    os.mkdir(basedir)