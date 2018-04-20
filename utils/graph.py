#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-20 上午10:25
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com
import tensorflow as tf
import numpy as np
#--------------------------------------------------#
#edge features
#ref:Dynamic Graph CNN for Learning on Point Clouds,arXiv preprint arXiv:1801.07829
def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)#增加维度

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])#交换输入张量的不同维度用的
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)#n*n
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)#这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
  return nn_idx

def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)#从tensor中删除所有大小是1的维度
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1])

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature

def edge_feature(point_cloud, k):
    '''input is BxNx3, default k=20'''
    adj_matrix = pairwise_distance(point_cloud)
    nn_idx = knn(adj_matrix, k=k)
    return get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
#---------------------------------------------------#


if __name__ == "__main__":
    #a = np.random.randint(0, 100, (10,10,3))
    x = tf.truncated_normal([10, 100, 3], dtype=tf.float32)
    b = edge_feature(x, 20)
    print(b)