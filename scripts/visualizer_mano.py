import argparse
import sys

from easydict import EasyDict

sys.path = ["."] + sys.path
import os.path as op
import numpy as np
from loguru import logger
import os

from common.viewer import ARCTICViewer, ViewerData
from common.xdict import xdict
import numpy as np
import torch
import trimesh
from common.body_models import build_layers, seal_mano_mesh
import common.viewer as viewer_utils
from smplx import MANO
from aitviewer.renderables.arrows import Arrows
from random import choice

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

    pose_r = torch.from_numpy(np.concatenate((raw_data['right_hand']['rot'], raw_data['right_hand']['pose']), axis=1))
    trans_r = torch.from_numpy(raw_data['right_hand']['trans'])

    trans_o = torch.from_numpy(raw_data[obj_key]['trans'])
    rot_o = torch.from_numpy(raw_data[obj_key]['rot'])

    device = 'cpu'
    MODEL_DIR = "./data/body_models/mano"

    mano_layer = MANO(
        MODEL_DIR,
        create_transl=False,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=True,
    )
    mano_layer = mano_layer.to(device)


    mano_r = mano_layer
    pose_r = pose_r.reshape(-1, 48)
    cam_r = trans_r.view(-1, 1, 3)

    num_frames = trans_o.shape[0]

    out_r = mano_r(
        global_orient=pose_r[:, :3].reshape(-1, 3),
        hand_pose=pose_r[:, 3:].reshape(-1, 45),
        betas=torch.zeros((num_frames, 10)).view(-1, 10),
    )

    v3d_r = out_r.vertices + cam_r

    f3d_r = torch.LongTensor(mano_layer.faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)

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
    v3d_r -= centers_o

    if 'points' in raw_data['right_hand']:
        line_origin = torch.from_numpy(raw_data['right_hand']['points'])[:, 1, :, :]
        line_tips = torch.from_numpy(raw_data['right_hand']['points'])[:, 0, :, :]
        rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        rotation = np.array(rotation).reshape(3,3)

        line_origin -= centers_o
        line_tips -= centers_o

        line_origin = np.matmul(line_origin, rotation.T)
        line_tips = np.matmul(line_tips, rotation.T)

        arrow = Arrows(line_origin.numpy(), line_tips.numpy(), r_base=0.002, r_head=0.004, color=(0.969, 0.106, 0.059, 1.0))
    else:
        arrow = None

    hand_color = "white"
    object_color = "light-blue"
    right = {
        "v3d": v3d_r.numpy(),
        "f3d": f3d_r.numpy(),
        "vc": None,
        "name": "right",
        "color": hand_color,
        'texture': None,
        'uv': None,
    }

    obj = {
        "v3d": v3d_o,
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": object_color,
        'texture': texture_p,
        'uv': uvs,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        {
            "right": right,
            "object": obj,
        },
        draw_edges=False,
        flat_shading=True,
    )

    meshes['object'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    meshes['right'].rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

    return meshes, num_frames, arrow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", type=str, default="")
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument("--zoom_out", type=float, default=0.5)
    parser.add_argument("--seq_name", type=str, default="WineGlass")
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
            meshes, num_frames, arrow = construct_meshes_per_object(obj_key, graspxl_data, set_color, obj_name, ambient_object=0.45, use_texture=use_texture)
            if len(meshes_all) > 0:
                pass
                #meshes.pop('table')
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
        return batch, arrow



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


    batch, arrow = viewer.load_data(data_p, texture, args.obj_name)
    viewer.render_seq(batch, out_folder=op.join(exp_folder, "render", seq_name), object_name=object_name, arrow=arrow)


if __name__ == "__main__":
    main()

