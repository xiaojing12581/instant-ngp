#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import code
#获取需求文件夹或文件路径列表
import glob
#读取和写入图像数据
import imageio
import numpy as np
import os
#PurePosixPath用于操作UNIX（包括Mac OSX）风格的路径，Path代表访问实际文件系统的真正路径
from pathlib import Path, PurePosixPath
#沿给定轴计算一维卷积，沿给定轴的阵列线与给定权重进行卷积
from scipy.ndimage.filters import convolve1d
#解决str和其他二进制数据类型的转换
import struct
import sys

import flip
import flip.utils
#path.resolve（）把一个路径或路径片段的序列解析为一个绝对路径，__file__表示当前的common.py文件，resolve()文件的绝对路径，parent.parent向上两层父级目录
PAPER_FOLDER = Path(__file__).resolve().parent.parent
SUPPL_FOLDER = PAPER_FOLDER/"supplemental"
SCRIPTS_FOLDER = PAPER_FOLDER/"scripts"
TEMPLATE_FOLDER = SCRIPTS_FOLDER/"template"
DATA_FOLDER = SCRIPTS_FOLDER/"data"

#os.path.realpath（）获得该方法所在脚本（.py文件）的路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
#os.environ.get（）：os.environ（获取有关系统的各种信息）是一个字典，是环境变量的字典，可以通过get获取键对应的值
NGP_DATA_FOLDER = os.environ.get("NGP_DATA_FOLDER") or os.path.join(ROOT_DIR, "data")


NERF_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "nerf")
SDF_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "sdf")
IMAGE_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "image")
VOLUME_DATA_FOLDER = os.path.join(NGP_DATA_FOLDER, "volume")

# Search for pyngp in the build folder.在构建文件夹中搜索pyngp。
#glob模块实现对目录内容进行匹配，结合通配符号*，？，[]，使用。*可以匹配零个或多个符号，？可以匹配单个字符，[]可以匹配一个指定范围的字符
#glob.glob（）函数接收通配模式做为输入，并返回所有匹配的文件名和路径名列表
#glob.iglob()获取一个可遍历对象，使用它可以逐个获取匹配的文件路径名，参数（文件名，recursive代表递归调用）
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

def repl(testbed):
	print("-------------------\npress Ctrl-Z to return to gui\n---------------------------")
	#按Ctrl-Z返回gui，交互式控制台
	code.InteractiveConsole(locals=locals()).interact()
	print("------- returning to gui...")

def mse2psnr(x): return -10.*np.log(x)/np.log(10.)

def sanitize_path(path):
	return str(PurePosixPath(path.relative_to(PAPER_FOLDER)))#path.relative_to计算相对路径

# from https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
def trapez(y,y0,w):
	return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
	# The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).如果c1 >= c0且c1-c0 >= abs(r1-r0)，则下面的算法工作正常。
	# If either of these cases are violated, do some switches.如果违反了这两种情况中的任何一种，请进行一些切换。
	if abs(c1-c0) < abs(r1-r0):
		# Switch x and y, and switch again when returning.切换x和y，返回时再次切换。
		xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
		return (yy, xx, val)

	# At this point we know that the distance in columns (x) is greater此时，我们知道列(x)中的距离大于行(y)中的距离。
	# than that in rows (y). Possibly one more switch if c0 > c1.如果c0 > c1，可能还有一个开关。
	if c0 > c1:
		return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

	# The following is now always < 1 in abs在abs中，以下值现在总是< 1
	slope = (r1-r0) / (c1-c0)

	# Adjust weight by the slope根据坡度调整重量
	w *= np.sqrt(1+np.abs(slope)) / 2

	# We write y as a function of x, because the slope is always <= 1我们把y写成x的函数，因为斜率总是< = 1
	# (in absolute value)
	x = np.arange(c0, c1+1, dtype=float)
	y = x * slope + (c1*r0-c0*r1) / (c1-c0)

	# Now instead of 2 values for y, we have 2*np.ceil(w/2).现在，我们用2*np.ceil(w/2)代替了y的2个值。
	# All values are 1 except the upmost and bottommost.除了最上面和最下面，所有值都是1。
	thickness = np.ceil(w/2)
	yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
	xx = np.repeat(x, yy.shape[1])
	vals = trapez(yy, y.reshape(-1,1), w).flatten()#flatten()降维，例如（2，3，4）维——>2*3*4维

	yy = yy.flatten()

	# Exclude useless parts and those outside of the interval排除无用部分和间隔之外的部分，以避免图片之外的部分
	# to avoid parts outside of the picture
	mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))#将一个数组中的元素逻辑与操作后得到一个结果
	#np.logical_and(x1,x2)两个布尔型数组逻辑与返回布尔型数组
	return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def diagonally_truncated_mask(shape, x_threshold, angle):#对角线_截断_掩码
	result = np.zeros(shape, dtype=bool)
	for x in range(shape[1]):
		for y in range(shape[0]):
			thres = x_threshold * shape[1] - (angle * shape[0] / 2) + y * angle
			result[y, x, ...] = x < thres
	return result

def diagonally_combine_two_images(img1, img2, x_threshold, angle, gap=0, color=1):#对角组合两幅图像
	if img2.shape != img1.shape:
		raise ValueError(f"img1 and img2 must have the same shape; {img1.shape} vs {img2.shape}")
	mask = diagonally_truncated_mask(img1.shape, x_threshold, angle)
	result = img2.copy()
	result[mask] = img1[mask]
	if gap > 0:
		rr, cc, val = weighted_line(0, int(x_threshold * img1.shape[1] - (angle * img1.shape[0] / 2)), img1.shape[0]-1, int(x_threshold * img1.shape[1] + (angle * img1.shape[0] / 2)), gap)
		result[rr, cc, :] = result[rr, cc, :] * (1 - val[...,np.newaxis]) + val[...,np.newaxis] * color#np.newaxis插入新维度，且扩张的那一维的长度是1
	return result

def diagonally_combine_images(images, x_thresholds, angle, gap=0, color=1):#对角组合图像
	result = images[0]
	for img, thres in zip(images[1:], x_thresholds):
		result = diagonally_combine_two_images(result, img, thres, angle, gap, color)
	return result

def write_image_imageio(img_file, img, quality):#写图像
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)#图像存储类型
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)#（要存储的路径，要保存的图片）保存图片到特定路径

def read_image_imageio(img_file):#读图像
	img = imageio.imread(img_file)
	img = np.asarray(img).astype(np.float32)#将数据结构转化为ndarray
	if len(img.shape) == 2:
		img = img[:,:,np.newaxis]
	return img / 255.0

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)#满足条件，输出前一个，不满足输出后一个

def linear_to_srgb(img):
	limit = 0.0031308
	#np.where满足条件输出前一个，不满足输出后一个
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
	if os.path.splitext(file)[1] == ".bin":
		with open(file, "rb") as f:#以二进制打开一个文件用于读取
			bytes = f.read()#将文件数据做为字符串返回
			#struct.pack(fmt,v1,v2...)将v1，v2等参数的值进行一层包装，包装的方法由fmt指定，被包装的参数必须严格符合fmt，最后返回一个包装后的字符串
			h, w = struct.unpack("ii", bytes[:8])#解包，返回一个由解包数据得到的一个元组
			#np.frombuffer将data以流的形式读入转化维ndarray对象。参数buffer缓冲区表示暴露缓冲区接口的对象
			#dtype代表返回的数据类型数组的数据类型；count代表返回ndarray的长度；offset偏移量，代表读取的起始位置
			img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
	else:
		img = read_image_imageio(file)
		if img.shape[2] == 4:
			img[...,0:3] = srgb_to_linear(img[...,0:3])
			# Premultiply alpha预乘α
			img[...,0:3] *= img[...,3:4]
		else:
			img = srgb_to_linear(img)
	return img

def write_image(file, img, quality=95):#写入图像
	if os.path.splitext(file)[1] == ".bin":
		if img.shape[2] < 4:
			#np.dstack将列表中的数组沿深度方向进行拼接
			img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
		with open(file, "wb") as f:
			f.write(struct.pack("ii", img.shape[0], img.shape[1]))
			f.write(img.astype(np.float16).tobytes())
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			# Unmultiply alpha取消阿尔法乘法
			#np.divide数组对应位置元素做除法，参数1充当被除数的数组，参数2充当除数的数组，out计算结果存放的位置，where数组型变量默认即可
			img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
			img[...,0:3] = linear_to_srgb(img[...,0:3])
		else:
			img = linear_to_srgb(img)
		write_image_imageio(file, img, quality)

def trim(error, skip=0.000001):#整齐
	error = np.sort(error.flatten())
	size = error.size
	skip = int(skip * size)
	return error[skip:size-skip].mean()

def luminance(a):#亮度
	return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def SSIM(a, b):
	def blur(a):#虚化
		k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
		x = convolve1d(a, k, axis=0)
		return convolve1d(x, k, axis=1)
	a = luminance(a)
	b = luminance(b)
	mA = blur(a)
	mB = blur(b)
	sA = blur(a*a) - mA**2
	sB = blur(b*b) - mB**2
	sAB = blur(a*b) - mA*mB
	c1 = 0.01**2
	c2 = 0.03**2
	p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
	p2 = (2.0*sAB + c2)/(sA + sB + c2)
	error = p1 * p2
	return error

def L1(img, ref):
	return np.abs(img - ref)

def APE(img, ref):
	return L1(img, ref) / (1e-2 + ref)

def SAPE(img, ref):
	return L1(img, ref) / (1e-2 + (ref + img) / 2.)

def L2(img, ref):
	return (img - ref)**2

def RSE(img, ref):
	return L2(img, ref) / (1e-2 + ref**2)

def rgb_mean(img):
	return np.mean(img, axis=2)

def compute_error_img(metric, img, ref):
	img[np.logical_not(np.isfinite(img))] = 0#np.isfinite逐元素测试有效性（不是无穷大，也不是非数字）结果以布尔数组的形式返回
	img = np.maximum(img, 0.)
	if metric == "MAE":
		return L1(img, ref)
	elif metric == "MAPE":
		return APE(img, ref)
	elif metric == "SMAPE":
		return SAPE(img, ref)
	elif metric == "MSE":
		return L2(img, ref)
	elif metric == "MScE":
		return L2(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric == "MRSE":
		return RSE(img, ref)
	elif metric == "MtRSE":
		return trim(RSE(img, ref))
	elif metric == "MRScE":
		return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
	elif metric == "SSIM":
		return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric in ["FLIP", "\FLIP"]:
		# Set viewing conditions设置查看条件
		monitor_distance = 0.7
		monitor_width = 0.7
		monitor_resolution_x = 3840
		# Compute number of pixels per degree of visual angle计算每度视角的像素数
		pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

		ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
		img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
		result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
		assert np.isfinite(result).all()
		return flip.utils.CHWtoHWC(result)

	raise ValueError(f"Unknown metric: {metric}.")

def compute_error(metric, img, ref):
	metric_map = compute_error_img(metric, img, ref)
	metric_map[np.logical_not(np.isfinite(metric_map))] = 0
	if len(metric_map.shape) == 3:
		metric_map = np.mean(metric_map, axis=2)
	mean = np.mean(metric_map)
	return mean
