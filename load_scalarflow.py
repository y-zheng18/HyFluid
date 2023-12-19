import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    ct = torch.Tensor([
        [1, 0, 0, wx],
        [0, 1, 0, wy],
        [0, 0, 1, wz],
        [0, 0, 0, 1]]).float()
    c2w = ct @ c2w

    return c2w


def load_pinf_frame_data(basedir, half_res=False, split='train'):
    # frame data
    all_imgs = []
    all_poses = []

    with open(os.path.join(basedir, 'info.json'), 'r') as fp:
        # read render settings
        meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]],
                              axis=1)  # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'], [3])

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num

        video_list = meta[split + '_videos'] if (split + '_videos') in meta else meta['train_videos'][0:1]

        for video_id, train_video in enumerate(video_list):
            imgs = []

            f_name = os.path.join(basedir, train_video['file_name'])
            reader = imageio.get_reader(f_name, "ffmpeg")
            for frame_i in range(train_video['frame_num']):
                reader.set_image_index(frame_i)
                frame = reader.get_next_data()

                H, W = frame.shape[:2]
                camera_angle_x = float(train_video['camera_angle_x'])
                Focal = .5 * W / np.tan(.5 * camera_angle_x)
                imgs.append(frame)

            reader.close()
            imgs = (np.float32(imgs) / 255.)

            if half_res:
                H = H // 2
                W = W // 2
                Focal = Focal / 2.

                imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
                for i, img in enumerate(imgs):
                    imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs = imgs_half_res

            all_imgs.append(imgs)
            all_poses.append(np.array(
                train_video['transform_matrix_list'][frame_i]
                if 'transform_matrix_list' in train_video else train_video['transform_matrix']
            ).astype(np.float32))

    imgs = np.stack(all_imgs, 0)  # [V, T, H, W, 3]
    imgs = np.transpose(imgs, [1, 0, 2, 3, 4])  # [T, V, H, W, 3]
    poses = np.stack(all_poses, 0)  # [V, 4, 4]
    hwf = np.float32([H, W, Focal])

    # set render settings:
    sp_n = 120  # an even number!
    sp_poses = [
        pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2])
        for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
    ]
    render_poses = torch.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps = np.arange(sp_n) / (sp_n - 1)

    return imgs, poses, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far
