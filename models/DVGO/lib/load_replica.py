'''DVGO.

The source code is adopted from:
https://github.com/sunset1995/DirectVoxGO

Reference:
[1] Sun C, Sun M, Chen H T.
    Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. IEEE/CVF Conference on Computer Vision and Pattern Recognition.
'''

import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import copy

def normalize(x):
    return x / np.linalg.norm(x)

def get_json_content(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_replica_data(basedir, movie_render_kwargs={}, skip_every_for_val_split=10):
    meta = get_json_content(os.path.join(basedir, 'meta_data.json'))
    indices = list(range(len(meta["frames"])))
    fx = []
    fy = []
    cx = []
    cy = []
    camera_to_worlds = []
    image_filenames = []
    i_split = []
    train_split = []
    val_split = []
    all_imgs = []
    all_poses = []
    for i, frame in enumerate(meta["frames"]):
        if (i) % skip_every_for_val_split:
            train_split.append(i)
        else:
            val_split.append(i)
        image_filename = os.path.join(basedir, frame["rgb_path"])

        all_imgs.append((imageio.imread(image_filename) / 255.).astype(np.float32))
        all_poses.append(np.array(frame["camtoworld"]).astype(np.float32))
    H, W = meta['height'], meta['width']
    K = np.eye(3)
    K[0, 0] = K[1, 1] = meta["frames"][0]['intrinsics'][0][0]
    K[0, 2] = meta["frames"][0]['intrinsics'][0][2]
    K[1, 2] = meta["frames"][0]['intrinsics'][1][2]
    focal = float(K[0,0])
    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    ### generate spiral poses for rendering fly-through movie
    centroid = poses[:,:3,3].mean(0)
    radcircle = movie_render_kwargs.get('scale_r', 1.0) * np.linalg.norm(poses[:,:3,3] - centroid, axis=-1).mean()
    centroid[0] += movie_render_kwargs.get('shift_x', 0)
    centroid[1] += movie_render_kwargs.get('shift_y', 0)
    centroid[2] += movie_render_kwargs.get('shift_z', 0)
    new_up_rad = movie_render_kwargs.get('pitch_deg', 0) * np.pi / 180
    target_y = radcircle * np.tan(new_up_rad)

    render_poses = []

    for th in np.linspace(0., 2.*np.pi, 200):
        camorigin = np.array([radcircle * np.cos(th), 0, radcircle * np.sin(th)])
        if movie_render_kwargs.get('flip_up_vec', False):
            up = np.array([0,-1.,0])
        else:
            up = np.array([0,1.,0])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin + centroid
        # rotate to align with new pitch rotation
        lookat = -vec2
        lookat[1] = target_y
        lookat = normalize(lookat)
        lookat *= -1
        vec2 = -lookat
        vec1 = normalize(np.cross(vec2, vec0))

        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_poses.append(p)
    i_split.append(train_split)
    i_split.append(val_split)
    i_split.append(val_split)
    render_poses = copy.deepcopy(poses)
    near, far = meta['scene_box']['near'], meta['scene_box']['far']
    poses[:, 0:3, 1:3] *= -1
    render_poses[:, 0:3, 1:3] *= -1
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far

if __name__ == '__main__':
    load_replica_data(basedir='data/Replica/scan4')

