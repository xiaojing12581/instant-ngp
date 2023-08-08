#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path

import numpy as np
import json
import copy
from pyquaternion import Quaternion#将旋转矩阵转换为四元数形式
from tqdm import tqdm
from PIL import Image

def rotate_img(img_path, degree=90):
	img = Image.open(img_path)
	img = img.rotate(degree, expand=1)#旋转图像
	img.save(img_path, quality=100, subsampling=0)

def rotate_camera(c2w, degree=90):
	rad = np.deg2rad(degree)#角度制转弧度制
	R = Quaternion(axis=[0, 0, -1], angle=rad)#旋转矩阵
	T = R.transformation_matrix#平移矩阵
	return c2w @ T#@一个装饰器，针对函数，起调用传参的作用

def swap_axes(c2w):
	rad = np.pi / 2
	R = Quaternion(axis=[1, 0, 0], angle=rad)
	T = R.transformation_matrix
	return T @ c2w#相机坐标系列到世界坐标系的转换

# Automatic rescale & offset the poses.自动重新缩放和偏移姿势
def find_transforms_center_and_scale(raw_transforms):
	print("computing center of attention...")#注意力的计算中心
	frames = raw_transforms['frames']
	for frame in frames:
		frame['transform_matrix'] = np.array(frame['transform_matrix'])

	rays_o = []
	rays_d = []
	for f in tqdm(frames):
		mf = f["transform_matrix"][0:3,:]
		rays_o.append(mf[:3,3:])
		rays_d.append(mf[:3,2:3])
	rays_o = np.asarray(rays_o)
	rays_d = np.asarray(rays_d)

	# Find the point that minimizes its distances to all rays.找到到所有光线的距离最小的点。
	def min_line_dist(rays_o, rays_d):
		A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])#np.eye生成对角阵，将一个label数组大小为(1,m)或(m,1)的数组转化成one-hot数组
		b_i = -A_i @ rays_o
		#删除单维、求逆、转置
		pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
		return pt_mindist

	translation = min_line_dist(rays_o, rays_d)
	normalized_transforms = copy.deepcopy(raw_transforms)#深拷贝，拷贝对象及其子对象
	for f in normalized_transforms["frames"]:
		f["transform_matrix"][0:3,3] -= translation

	# Find the scale.求尺度
	avglen = 0.
	for f in normalized_transforms["frames"]:
		avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
	nframes = len(normalized_transforms["frames"])
	avglen /= nframes
	print("avg camera distance from origin", avglen)#从原点到相机的平均距离
	scale = 4.0 / avglen # scale to "nerf sized"缩放至“nerf大小”

	return translation, scale

def normalize_transforms(transforms, translation, scale):
	normalized_transforms = copy.deepcopy(transforms)
	for f in normalized_transforms["frames"]:
		f["transform_matrix"] = np.asarray(f["transform_matrix"])
		f["transform_matrix"][0:3,3] -= translation
		f["transform_matrix"][0:3,3] *= scale
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return normalized_transforms

def parse_args():
	#将Record3D捕获转换为nerf格式transforms.json
	parser = argparse.ArgumentParser(description="convert a Record3D capture to nerf format transforms.json")
	parser.add_argument("--scene", default="", help="path to the Record3D capture")#记录3D捕获的路径
	parser.add_argument("--rotate", action="store_true", help="rotate the dataset")#旋转数据集
	parser.add_argument("--subsample", default=1, type=int, help="step size of subsampling")#子采样的步长
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	dataset_dir = Path(args.scene)
	with open(dataset_dir / 'metadata') as f:
		metadata = json.load(f)

	frames = []
	n_images = len(list((dataset_dir / 'rgbd').glob('*.jpg')))
	poses = np.array(metadata['poses'])
	for idx in tqdm(range(n_images)):
		# Link the image.链接图像
		img_name = f'{idx}.jpg'
		img_path = dataset_dir / 'rgbd' / img_name

		# Rotate the image.旋转图像
		if args.rotate:
			# TODO: parallelize this step with joblib. TODO:使用joblib并行执行此步骤
			rotate_img(img_path)

		# Extract c2w.提取c2w。
		""" Each `pose` is a 7-element tuple which contains quaternion + world position.
  			每个“姿势”是一个7元素的元组，它包含四元数+世界位置。
			[qx, qy, qz, qw, tx, ty, tz]
		"""
		pose = poses[idx]
		q = Quaternion(x=pose[0], y=pose[1], z=pose[2], w=pose[3])
		c2w = np.eye(4)
		c2w[:3, :3] = q.rotation_matrix
		c2w[:3, -1] = [pose[4], pose[5], pose[6]]
		if args.rotate:
			c2w = rotate_camera(c2w)
			c2w = swap_axes(c2w)

		frames.append(
			{
				"file_path": f"./rgbd/{img_name}",
				"transform_matrix": c2w.tolist(),
			}
		)

	# Write intrinsics to `cameras.txt`.将内部函数写入“cameras.txt”。
	if not args.rotate:
		h = metadata['h']
		w = metadata['w']
		K = np.array(metadata['K']).reshape([3, 3]).T
		fx = K[0, 0]
		fy = K[1, 1]
		cx = K[0, 2]
		cy = K[1, 2]
	else:
		h = metadata['w']
		w = metadata['h']
		K = np.array(metadata['K']).reshape([3, 3]).T
		fx = K[1, 1]
		fy = K[0, 0]
		cx = K[1, 2]
		cy = h - K[0, 2]

	transforms = {}
	transforms['fl_x'] = fx
	transforms['fl_y'] = fy
	transforms['cx'] = cx
	transforms['cy'] = cy
	transforms['w'] = w
	transforms['h'] = h
	transforms['aabb_scale'] = 16
	transforms['scale'] = 1.0
	transforms['camera_angle_x'] = 2 * np.arctan(transforms['w'] / (2 * transforms['fl_x']))
	transforms['camera_angle_y'] = 2 * np.arctan(transforms['h'] / (2 * transforms['fl_y']))
	transforms['frames'] = frames

	os.makedirs(dataset_dir / 'arkit_transforms', exist_ok=True)
	with open(dataset_dir / 'arkit_transforms' / 'transforms.json', 'w') as fp:
		json.dump(transforms, fp, indent=2)

	# Normalize the poses.使姿势正常化。
	transforms['frames'] = transforms['frames'][::args.subsample]
	translation, scale = find_transforms_center_and_scale(transforms)
	normalized_transforms = normalize_transforms(transforms, translation, scale)

	output_path = dataset_dir / 'transforms.json'
	with open(output_path, "w") as outfile:
		json.dump(normalized_transforms, outfile, indent=2)
