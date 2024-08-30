import argparse
import sys

from easydict import EasyDict

sys.path = ["."] + sys.path
import os.path as op
import numpy as np
from loguru import logger

from common.viewer import ARCTICViewer, ViewerData
from common.xdict import xdict
import numpy as np
import torch

import trimesh
import common.viewer as viewer_utils

from urdfpy import URDF
import pyrender

def axisangle2mat(
        rot_vecs,
        epsilon: float = 1e-8,
):
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat.numpy()

def construct_meshes_per_object(obj_key, raw_data, set_color, obj_name, ambient_object, use_texture=False):
    trans_r = torch.from_numpy(raw_data['right_hand']['trans'])
    rot_r = torch.from_numpy(raw_data['right_hand']['rot'])
    pose_r = torch.from_numpy(raw_data['right_hand']['pose'])

    trans_o = torch.from_numpy(raw_data[obj_key]['trans'])
    rot_o = torch.from_numpy(raw_data[obj_key]['rot'])

    num_frames = trans_r.shape[0]

    robot = URDF.load("data/body_models/allegro/allegro.urdf")

    joint_list = ["x_joint", "y_joint", "z_joint", "x_rotation_joint", "y_rotation_joint", "z_rotation_joint", "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0", "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0", "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0", "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"]

    v3d_r1 = np.zeros((num_frames, 1606, 3))
    v3d_r2 = np.zeros((num_frames, 2044, 3))
    v3d_r3 = np.zeros((num_frames, 577, 3))
    v3d_r4 = np.zeros((num_frames, 577, 3))
    v3d_r5 = np.zeros((num_frames, 577, 3))
    v3d_r6 = np.zeros((num_frames, 226, 3))
    v3d_r7 = np.zeros((num_frames, 702, 3))
    v3d_r8 = np.zeros((num_frames, 702, 3))
    v3d_r9 = np.zeros((num_frames, 702, 3))
    v3d_r10 = np.zeros((num_frames, 470, 3))
    v3d_r11 = np.zeros((num_frames, 530, 3))
    v3d_r12 = np.zeros((num_frames, 530, 3))
    v3d_r13 = np.zeros((num_frames, 530, 3))
    v3d_r14 = np.zeros((num_frames, 434, 3))
    v3d_r15 = np.zeros((num_frames, 180, 3))
    v3d_r16 = np.zeros((num_frames, 180, 3))
    v3d_r17 = np.zeros((num_frames, 180, 3))
    v3d_r18 = np.zeros((num_frames, 155, 3))
    v3d_r19 = np.zeros((num_frames, 155, 3))
    v3d_r20 = np.zeros((num_frames, 155, 3))
    v3d_r21 = np.zeros((num_frames, 155, 3))

    for i in range(num_frames):
        link_list = []
        pose_list = []
        current_cfg = {}
        for j in range(len(joint_list)):
            current_cfg[joint_list[j]] = pose_r[i, j]

        fk = robot.visual_trimesh_fk(cfg=current_cfg)
        for tm in fk:
            link_list.append(tm)
            pose_list.append(fk[tm])

        rot_mat = axisangle2mat(-rot_r)

        f3d_r1 = link_list[0].faces
        temp_verts = np.ones((link_list[0].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[0].vertices
        temp_verts = np.matmul(temp_verts, pose_list[0].T)[:, :3]
        v3d_r1[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r2 = link_list[1].faces
        temp_verts = np.ones((link_list[1].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[1].vertices
        temp_verts = np.matmul(temp_verts, pose_list[1].T)[:, :3]
        v3d_r2[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r3 = link_list[2].faces
        temp_verts = np.ones((link_list[2].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[2].vertices
        temp_verts = np.matmul(temp_verts, pose_list[2].T)[:, :3]
        v3d_r3[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r4 = link_list[3].faces
        temp_verts = np.ones((link_list[3].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[3].vertices
        temp_verts = np.matmul(temp_verts, pose_list[3].T)[:, :3]
        v3d_r4[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r5 = link_list[4].faces
        temp_verts = np.ones((link_list[4].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[4].vertices
        temp_verts = np.matmul(temp_verts, pose_list[4].T)[:, :3]
        v3d_r5[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r6 = link_list[5].faces
        temp_verts = np.ones((link_list[5].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[5].vertices
        temp_verts = np.matmul(temp_verts, pose_list[5].T)[:, :3]
        v3d_r6[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r7 = link_list[6].faces
        temp_verts = np.ones((link_list[6].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[6].vertices
        temp_verts = np.matmul(temp_verts, pose_list[6].T)[:, :3]
        v3d_r7[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r8 = link_list[7].faces
        temp_verts = np.ones((link_list[7].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[7].vertices
        temp_verts = np.matmul(temp_verts, pose_list[7].T)[:, :3]
        v3d_r8[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r9 = link_list[8].faces
        temp_verts = np.ones((link_list[8].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[8].vertices
        temp_verts = np.matmul(temp_verts, pose_list[8].T)[:, :3]
        v3d_r9[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r10 = link_list[9].faces
        temp_verts = np.ones((link_list[9].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[9].vertices
        temp_verts = np.matmul(temp_verts, pose_list[9].T)[:, :3]
        v3d_r10[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r11 = link_list[10].faces
        temp_verts = np.ones((link_list[10].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[10].vertices
        temp_verts = np.matmul(temp_verts, pose_list[10].T)[:, :3]
        v3d_r11[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r12 = link_list[11].faces
        temp_verts = np.ones((link_list[11].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[11].vertices
        temp_verts = np.matmul(temp_verts, pose_list[11].T)[:, :3]
        v3d_r12[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r13 = link_list[12].faces
        temp_verts = np.ones((link_list[12].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[12].vertices
        temp_verts = np.matmul(temp_verts, pose_list[12].T)[:, :3]
        v3d_r13[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r14 = link_list[13].faces
        temp_verts = np.ones((link_list[13].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[13].vertices
        temp_verts = np.matmul(temp_verts, pose_list[13].T)[:, :3]
        v3d_r14[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r15 = link_list[14].faces
        temp_verts = np.ones((link_list[14].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[14].vertices
        temp_verts = np.matmul(temp_verts, pose_list[14].T)[:, :3]
        v3d_r15[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r16 = link_list[15].faces
        temp_verts = np.ones((link_list[15].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[15].vertices
        temp_verts = np.matmul(temp_verts, pose_list[15].T)[:, :3]
        v3d_r16[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r17 = link_list[16].faces
        temp_verts = np.ones((link_list[16].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[16].vertices
        temp_verts = np.matmul(temp_verts, pose_list[16].T)[:, :3]
        v3d_r17[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r18 = link_list[17].faces
        temp_verts = np.ones((link_list[17].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[17].vertices
        temp_verts = np.matmul(temp_verts, pose_list[17].T)[:, :3]
        v3d_r18[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r19 = link_list[18].faces
        temp_verts = np.ones((link_list[18].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[18].vertices
        temp_verts = np.matmul(temp_verts, pose_list[18].T)[:, :3]
        v3d_r19[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r20 = link_list[19].faces
        temp_verts = np.ones((link_list[19].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[19].vertices
        temp_verts = np.matmul(temp_verts, pose_list[19].T)[:, :3]
        v3d_r20[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))

        f3d_r21 = link_list[20].faces
        temp_verts = np.ones((link_list[20].vertices.shape[0], 4))
        temp_verts[:, :3] = link_list[20].vertices
        temp_verts = np.matmul(temp_verts, pose_list[20].T)[:, :3]
        v3d_r21[i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_r.numpy()[i, :].reshape(1, 3), (temp_verts.shape[0], 1))


    mesh_p = f"./data/GraspXL/object_mesh/{obj_name}.obj"
    mesh = trimesh.load(mesh_p, process=False)


    texture_p = None
    uvs = None

    object_v = mesh.vertices
    f3d_o = mesh.faces

    frame_num = trans_o.shape[0]

    v3d_o = np.zeros((frame_num, object_v.shape[0], 3))
    for i in range(frame_num):
        rot_mat = axisangle2mat(-rot_o)
        v3d_o[i, :, :] = np.matmul(object_v, rot_mat[i]) + np.tile(trans_o.numpy()[i, :].reshape(1, 3), (object_v.shape[0], 1))

    centers_o = torch.from_numpy(v3d_o).mean(dim=1, keepdim=True)
    centers_o = centers_o.mean(dim=0, keepdim=True)
    v3d_o -= centers_o.numpy() # frame x verts x 3
    v3d_r1 -= centers_o.numpy()
    v3d_r2 -= centers_o.numpy()
    v3d_r3 -= centers_o.numpy()
    v3d_r4 -= centers_o.numpy()
    v3d_r5 -= centers_o.numpy()
    v3d_r6 -= centers_o.numpy()
    v3d_r7 -= centers_o.numpy()
    v3d_r8 -= centers_o.numpy()
    v3d_r9 -= centers_o.numpy()
    v3d_r10 -= centers_o.numpy()
    v3d_r11 -= centers_o.numpy()
    v3d_r12 -= centers_o.numpy()
    v3d_r13 -= centers_o.numpy()
    v3d_r14 -= centers_o.numpy()
    v3d_r15 -= centers_o.numpy()
    v3d_r16 -= centers_o.numpy()
    v3d_r17 -= centers_o.numpy()
    v3d_r18 -= centers_o.numpy()
    v3d_r19 -= centers_o.numpy()
    v3d_r20 -= centers_o.numpy()
    v3d_r21 -= centers_o.numpy()

    # AIT meshes
    hand_color = "mixer"
    object_color = "light-blue"

    #
    right1 = {
        "v3d": v3d_r1,
        "f3d": f3d_r1,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right2 = {
        "v3d": v3d_r2,
        "f3d": f3d_r2,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right3 = {
        "v3d": v3d_r3,
        "f3d": f3d_r3,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right4 = {
        "v3d": v3d_r4,
        "f3d": f3d_r4,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right5 = {
        "v3d": v3d_r5,
        "f3d": f3d_r5,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right6 = {
        "v3d": v3d_r6,
        "f3d": f3d_r6,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right7 = {
        "v3d": v3d_r7,
        "f3d": f3d_r7,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right8 = {
        "v3d": v3d_r8,
        "f3d": f3d_r8,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right9 = {
        "v3d": v3d_r9,
        "f3d": f3d_r9,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right10 = {
        "v3d": v3d_r10,
        "f3d": f3d_r10,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right11 = {
        "v3d": v3d_r11,
        "f3d": f3d_r11,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right12 = {
        "v3d": v3d_r12,
        "f3d": f3d_r12,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right13 = {
        "v3d": v3d_r13,
        "f3d": f3d_r13,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right14 = {
        "v3d": v3d_r14,
        "f3d": f3d_r14,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right15 = {
        "v3d": v3d_r15,
        "f3d": f3d_r15,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right16 = {
        "v3d": v3d_r16,
        "f3d": f3d_r16,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right17 = {
        "v3d": v3d_r17,
        "f3d": f3d_r17,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right18 = {
        "v3d": v3d_r18,
        "f3d": f3d_r18,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right19 = {
        "v3d": v3d_r19,
        "f3d": f3d_r19,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right20 = {
        "v3d": v3d_r20,
        "f3d": f3d_r20,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }

    right21 = {
        "v3d": v3d_r21,
        "f3d": f3d_r21,
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
        # 'ambient': 0.5,
    }




    obj = {
        "v3d": v3d_o,
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": object_color,
        'texture': texture_p,
        'uv': uvs,
        #'ambient': ambient_object
    }

    meshes = viewer_utils.construct_viewer_meshes(
        {
            "right1": right1,
            "right2": right2,
            "right3": right3,
            "right4": right4,
            "right5": right5,
            "right6": right6,
            "right7": right7,
            "right8": right8,
            "right9": right9,
            "right10": right10,
            "right11": right11,
            "right12": right12,
            "right13": right13,
            "right14": right14,
            "right15": right15,
            "right16": right16,
            "right17": right17,
            "right18": right18,
            "right19": right19,
            "right20": right20,
            "right21": right21,
            "object": obj,
        },
        draw_edges=False,
        flat_shading=True,
    )

    meshes['object'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right1'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right2'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right3'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right4'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right5'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right6'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right7'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right8'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right9'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right10'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right11'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right12'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right13'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right14'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right15'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right16'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right17'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right18'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right19'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right20'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right21'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

    return meshes, num_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", type=str, default="")
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--zoom_out", type=float, default=0.5)
    parser.add_argument("--seq_name", type=str, default="allegro_WineGlass")
    parser.add_argument("--obj_name", type=str, default="wineglass")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--texture", action="store_true")
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args

class GraspXLViewer(ARCTICViewer):
    def load_data(self, data_p, use_texture, obj_name):
        logger.info("Creating meshes")

        graspxl_data = np.load(data_p, allow_pickle=True).item()

        object_keys = list(graspxl_data.keys())
        object_keys.remove('right_hand')

        # load mesh
        meshes_all = xdict()

        for obj_key in object_keys:
            set_color = 'box'
            meshes, num_frames = construct_meshes_per_object(obj_key, graspxl_data, set_color, obj_name, ambient_object=0.45, use_texture=use_texture)
            if len(meshes_all) > 0:
                pass
            if 'right' in meshes_all:
                meshes.pop('right')

            obj_mesh = meshes.pop('object')
            obj_mesh.name = obj_key
            meshes[obj_key] = obj_mesh
            meshes_all.merge(meshes)


        # setup camera
        focal = 1000.0
        rows = 224
        cols = 224
        K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
        Rt = np.zeros((num_frames, 3, 4))
        Rt[:, :3, :3] = np.eye(3)
        Rt[:, 1:3, :3] *= -1.0

        # pack data
        data = ViewerData(Rt=Rt, K=K, cols=cols, rows=rows, imgnames=None)
        batch = meshes_all, data
        self.check_format(batch)
        logger.info("Done")
        return batch



def main():
    args = parse_args()
    exp_folder = args.exp_folder
    seq_name = args.seq_name
    texture = args.texture
    viewer = GraspXLViewer(
        interactive=not args.headless,
        size=(8024, 4024),
        render_types=["rgb"],
    )
    logger.info(f"Rendering {seq_name}")
    data_p = f'./data/GraspXL/recorded/{seq_name}.npy'
    object_name = data_p.split('/')[-1].split('.')[0]

    batch = viewer.load_data(data_p, texture, args.obj_name)
    viewer.render_seq(batch, out_folder=op.join(exp_folder, "render", seq_name), object_name=object_name)


if __name__ == "__main__":
    main()

