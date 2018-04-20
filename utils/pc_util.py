#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-2 上午11:36
# @Author  : Yewei Li
# @E-mail  : liyewei20@163.com

import numpy as np
from utils.plyfile import PlyData,PlyElement
# Draw point cloud
from utils.eulerangles import euler2mat

#---------------------------Point cloud IO-------------------------#
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
#---------------------------------------------------------------------#


#------------------Draw point cloud------------------------#
def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image

def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large
#-----------------------------------------------------------#

#----------------------------Point cloud augument---------------------#
def shuffle_data(data, labels):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,... numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  return data[idx, ...], labels[idx], idx

def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data
#-----------------------------------------------------------------------------#