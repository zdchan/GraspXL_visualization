import argparse
import sys

sys.path = ["."] + sys.path
import os.path as op

import numpy as np
import torch
import trimesh
from easydict import EasyDict
from loguru import logger
from urdfpy import URDF

import common.viewer as viewer_utils
from common.viewer import ARCTICViewer, ViewerData
from common.xdict import xdict

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

    link_vert_counts = [1606, 2044, 577, 577, 577, 226, 702, 702, 702, 470, 530, 530, 530, 434, 180, 180, 180, 155, 155, 155, 155]
    num_links = len(link_vert_counts)

    v3d_links = [np.zeros((num_frames, vc, 3)) for vc in link_vert_counts]
    f3d_links = [None] * num_links

    for i in range(num_frames):
        current_cfg = {joint_list[j]: pose_r[i, j] for j in range(len(joint_list))}

        fk = robot.visual_trimesh_fk(cfg=current_cfg)
        link_list = list(fk.keys())
        pose_list = list(fk.values())

        rot_mat = axisangle2mat(-rot_r)
        trans_np = trans_r.numpy()[i, :].reshape(1, 3)

        for k in range(num_links):
            f3d_links[k] = link_list[k].faces
            temp_verts = np.ones((link_list[k].vertices.shape[0], 4))
            temp_verts[:, :3] = link_list[k].vertices
            temp_verts = np.matmul(temp_verts, pose_list[k].T)[:, :3]
            v3d_links[k][i, :, :] = np.matmul(temp_verts, rot_mat[i]) + np.tile(trans_np, (temp_verts.shape[0], 1))


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
    v3d_o -= centers_o.numpy()
    for k in range(num_links):
        v3d_links[k] -= centers_o.numpy()

    # AIT meshes
    hand_color = "mixer"
    object_color = "light-blue"

    mesh_dict = {}
    for k in range(num_links):
        mesh_dict[f"right{k+1}"] = {
            "v3d": v3d_links[k],
            "f3d": f3d_links[k],
            "vc": None,
            "name": "right",
            "color": hand_color,
            'texture': None,
            'uv': None,
        }

    mesh_dict["object"] = {
        "v3d": v3d_o,
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": object_color,
        'texture': texture_p,
        'uv': uvs,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        mesh_dict,
        draw_edges=False,
        flat_shading=True,
    )

    rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['object'].rotation = rotation
    for k in range(num_links):
        meshes[f'right{k+1}'].rotation = rotation

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

