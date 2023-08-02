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
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

def parse_args():
	#使用附加配置和输出选项运行即时神经图形图元
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")
	#要加载的文件。可以是场景、网络配置、快照、摄像机路径或它们的组合
	parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")
	#要加载的场景。可以是场景的名称或训练数据的完整路径。可以是NeRF数据集，a *。obj/*。用于训练SDF、图像或*的stl网格。nvdb卷
	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	#阻止参数显示在默认帮助输出中
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	#网络配置的路径。如果未指定，则使用场景的默认值
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")
	#训练前加载此快照。推荐的扩展名:。ingp/。msgpack
	parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	#训练后保存此快照。推荐的扩展名:。ingp/。msgpack
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")
	#将参数与原始NeRF匹配。在某些场景中会导致速度变慢和效果变差，但有助于合成场景中的高PSNR
	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	#一个nerf风格的路径转换json，我们将从其中计算PSNR
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	#为nerf设置训练光线开始时与摄影机的距离。< 0表示使用ngp默认值
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	#控制图像的亮度。正数增加亮度，负数降低亮度
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	#用于保存屏幕截图的nerf style transforms.json的路径
	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	#要对哪些帧进行截图
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	#截图输出到哪个目录
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	#截图中每个像素的样本数
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	#要渲染的相机路径，例如base_cam.json
	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	#对相机轨迹应用额外的平滑，但警告可能无法到达相机路径的端点。
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	#每秒帧数
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	#渲染的视频应该有多长的秒数
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	#将输出限制在START_FRAME和END_FRAME之间的帧
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	#每个像素的样本数。数字越大，噪波越少，但渲染速度越慢
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	#输出视频(video.mp4)或视频帧(video_%%04d.png)的文件名
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

	#从NeRF或SDF模型输出基于行进立方体的网格。支持OBJ和PLY格式
	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	#设置行进立方体网格的分辨率
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	#设置行进立方体的密度阈值
	parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

	#GUI和屏幕截图的分辨率宽度
	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	#GUI和屏幕截图的分辨率高度
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	#交互式运行测试平台GUI
	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	#如果启用了GUI，则控制是否立即开始培训
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	#放弃前要训练的步数
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	#打开包含主输出副本的第二个窗口
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	#渲染到虚拟现实耳机
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	#设置应用于NeRF训练图像的锐化量。范围从0.0到1.0
	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")


	return parser.parse_args()

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

if __name__ == "__main__":
	args = parse_args()
	if args.vr: # VR implies having the GUI running at the moment
		args.gui = True

	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	for file in args.files:
		scene_info = get_scene(file)
		if scene_info:
			file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
		testbed.load_file(file)

	if args.scene:
		scene_info = get_scene(args.scene)
		if scene_info is not None:
			args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
			if not args.network and "network" in scene_info:
				args.network = scene_info["network"]

		testbed.load_training_data(args.scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window)
		if args.vr:
			testbed.init_vr()


	if args.load_snapshot:
		scene_info = get_scene(args.load_snapshot)
		if scene_info is not None:
			args.load_snapshot = default_snapshot_filename(scene_info)
		testbed.load_snapshot(args.load_snapshot)
	elif args.network:
		testbed.reload_network_from_file(args.network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_lens_distortion = True

	network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
	if testbed.mode == ngp.TestbedMode.Sdf:
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Match nerf paper behaviour and train on a fixed bg.
		testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="steps") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_snapshot:
		os.makedirs(os.path.dirname(args.save_snapshot), exist_ok=True)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.render_min_transmittance = 1e-4

		testbed.shall_train = False
		testbed.load_training_data(args.test_transforms)

		with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
			for i in t:
				resolution = testbed.nerf.training.dataset.metadata[i].resolution
				testbed.render_ground_truth = True
				testbed.set_camera_to_training_view(i)
				ref_image = testbed.render(resolution[0], resolution[1], 1, True)
				testbed.render_ground_truth = False
				image = testbed.render(resolution[0], resolution[1], spp, True)

				if i == 0:
					write_image(f"ref.png", ref_image)
					write_image(f"out.png", image)

					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0
					write_image("diff.png", diffimg)

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		thresh = args.marching_cubes_density_thresh or 2.5
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)

	if ref_transforms:
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f.get("transform_matrix", f["transform_matrix_start"])
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)

		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps
		save_frames = "%" in args.video_output
		start_frame, end_frame = args.video_render_range

		if "tmp" in os.listdir():
			shutil.rmtree("tmp")
		os.makedirs("tmp")

		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			testbed.camera_smoothing = args.video_camera_smoothing

			if start_frame >= 0 and i < start_frame:
				# For camera smoothing and motion blur to work, we cannot just start rendering
				# from middle of the sequence. Instead we render a very small image and discard it
				# for these initial frames.
				# TODO Replace this with a no-op render method once it's available
				frame = testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
				continue
			elif end_frame >= 0 and i > end_frame:
				continue

			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
			if save_frames:
				write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
			else:
				write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

		if not save_frames:
			os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")

		shutil.rmtree("tmp")
