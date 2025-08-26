import cv2
import numpy as np
import pyrealsense2 as rs
import os
import json
import utils
import shutil

# 读取 Aruco Board 配置
config_path = os.path.join(os.getcwd(), 'configs', 'aruco', 'sl200_ml120.json')
with open(config_path, 'r') as f:
    cfg = json.load(f)

sl = cfg['sl']
ml = cfg['ml']
ids = cfg['ids']
charucodict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((3, 3), sl, ml, charucodict, np.array([ids]))

axis_offset = -int(sl * 3 / 2)
origin_offset = np.array([[axis_offset, axis_offset, 0]])
Rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

x, y, z, c = 100, 100, 200, 0
bbox3d = None

# Realsense 配置
pipe = rs.pipeline()
rscfg = rs.config()
width, height = 1280, 720
rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
profile = pipe.start(rscfg)

# 获取 Realsense 内参
intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K_full = np.array([[intrin.fx, 0, intrin.ppx], [0, intrin.fy, intrin.ppy], [0, 0, 1]])

# 创建数据保存目录
dataset_name = "penguin"
dataset_pth = os.path.join(os.getcwd(), 'out', 'datasets', dataset_name)
if os.path.exists(dataset_pth):
    shutil.rmtree(dataset_pth)
os.makedirs(dataset_pth)
os.makedirs(os.path.join(dataset_pth, 'color'))
os.makedirs(os.path.join(dataset_pth, 'raw_color'))
os.makedirs(os.path.join(dataset_pth, 'intrin_ba'))
os.makedirs(os.path.join(dataset_pth, 'poses_ba'))
os.makedirs(os.path.join(dataset_pth, 'reproj_box'))
os.makedirs(os.path.join(dataset_pth, 'raw_depth'))
os.makedirs(os.path.join(dataset_pth, 'bbox3d'))  # 新增目录用于保存3D边界框坐标

# 视频录制设置
frame_interval = 2      # 每 5 帧保存一次
frame_count    = 0
idx            = 0
recording      = False  # 录制状态

video_path     = os.path.join(dataset_pth, 'video.avi')
raw_video_path = os.path.join(dataset_pth, 'raw_video.avi')
fourcc         = cv2.VideoWriter_fourcc(*'XVID')
fps            = 15

video_writer     = cv2.VideoWriter(video_path,     fourcc, fps, (512, 512))
raw_video_writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (512, 512))

while True:
    # 定义3D边界框的8个顶点坐标
    bbox3d = np.array([
        [x, y, c], [x, y, z + c], [x, -y, z + c], [x, -y, c],
        [-x, y, c], [-x, y, z + c], [-x, -y, z + c], [-x, -y, c]
    ])

    # 读取帧
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    bgr = np.asanyarray(color_frame.get_data())
    frame_count += 1

    # 处理裁剪后的图像
    bbox = np.array([int(width / 2 - height / 2), 0, width - int(width / 2 - height / 2), height])
    bgr_crop, K_crop    = utils.crop_img_by_bbox(bgr, bbox, K=K_full, crop_size = 512)
    bgr_crop_raw, K_raw = utils.crop_img_by_bbox(bgr, bbox, K=K_full, crop_size = 512)
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(bgr_crop, charucodict)
    cv2.aruco.drawDetectedMarkers(bgr_crop, markerCorners, markerIds)

    if markerIds is not None:
        objPoints, imgPoints = board.matchImagePoints(markerCorners, markerIds)
        objPoints = objPoints + origin_offset
        _, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K_crop, distCoeffs=None)
        rmat = cv2.Rodrigues(rvec)[0] @ Rx180
        rvec = cv2.Rodrigues(rmat)[0]

        pose = np.eye(4)
        pose[:3, :3] = rmat
        pose[:3, 3] = tvec[:, 0]
        
        # 将3D边界框顶点转换到相机坐标系
        bbox3d_cam = (pose @ np.vstack([bbox3d.T, np.ones(8)]))[:3].T
        
        # 投影到2D图像
        bbox2d = utils.reproj(K_crop, pose, bbox3d)
        utils.draw_3d_box(bgr_crop, bbox2d)
        cv2.drawFrameAxes(bgr_crop, K_crop, None, rvec, tvec, 100)

        # 录制时保存数据
        if recording and frame_count % frame_interval == 0:
            print(f'保存数据，帧 {idx}')
            
            # 未标注图片
            bgr_crop_clean, K_crop_clean = utils.crop_img_by_bbox(bgr, bbox, K=K_full, crop_size = 512)
            cv2.imwrite(os.path.join(dataset_pth, 'color', f'{idx}.png'), bgr_crop_clean)
            

            # 标注图片
            cv2.imwrite(os.path.join(dataset_pth, 'color_with_pose', f'{idx}.png'), bgr_crop)
            
            # 标注数据
            np.savetxt(os.path.join(dataset_pth, 'intrin_ba', f'{idx}.txt'), K_crop)
            np.savetxt(os.path.join(dataset_pth, 'poses_ba', f'{idx}.txt'), pose)
            np.savetxt(os.path.join(dataset_pth, 'reproj_box', f'{idx}.txt'), bbox2d)
            
            # 保存3D边界框顶点坐标（相机坐标系）
            np.savetxt(os.path.join(dataset_pth, 'bbox3d', f'{idx}.txt'), bbox3d_cam)
            
            # 录制视频
            video_writer.write(bgr_crop)
            
            # 保存未标注视频
            raw_video_writer.write(bgr_crop_raw)
            
            idx += 1

    # 显示处理后的画面
    cv2.imshow('frame_crop', bgr_crop)
    key = cv2.waitKey(1)

    # 按键交互
    if key == ord('q'):
        break
    elif key == ord('w'):
        origin_offset[0, 2] += 10
        c -= 10
    elif key == ord('s'):
        origin_offset[0, 2] -= 10
        c += 10
    elif key == ord('e'):
        x += 10
    elif key == ord('d'):
        x -= 10
    elif key == ord('r'):
        y += 10
    elif key == ord('f'):
        y -= 10
    elif key == ord('t'):
        z += 10
    elif key == ord('g'):
        z -= 10
    elif key == 13:  # 回车键
        recording = not recording
        print("开始录制..." if recording else "暂停录制...")

# 释放资源
pipe.stop()
video_writer.release()
raw_video_writer.release()
cv2.destroyAllWindows()

# 保存3D边界框的原始定义（世界坐标系）
np.savetxt(os.path.join(dataset_pth, 'box3d_corners.txt'), bbox3d)
print("录制完成，数据已保存。")