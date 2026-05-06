# -*- coding: utf-8 -*-
"""
Visualizer for large-format GraspXL recorded data.

Expected sequence layout:
    {data_root}/{collection}/{object_dict}/{object_name}/{hand_model}_{num_id}.npy

Where collection follows:
    {hand_model}_{group}

Object meshes are resolved from:
    {object_mesh_root}/{object_dict}/{object_name}/{object_name}.obj

Hand models are resolved from:
    {body_model_root}/{hand_model}/
"""
import argparse
import glob
import json
import os
import os.path as op
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path = ["."] + sys.path

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    import common.viewer as viewer_utils
    from common.viewer import ARCTICViewer, ViewerData
except ImportError:
    viewer_utils = None
    ARCTICViewer = object
    ViewerData = None


DEFAULT_DATA_ROOT = "./data/GraspXL/recorded"
DEFAULT_OBJECT_MESH_ROOT = "./data/GraspXL/object_mesh"
DEFAULT_BODY_MODEL_ROOT = "./data/body_models"

VIRTUAL_JOINTS = [
    "x_joint",
    "y_joint",
    "z_joint",
    "x_rotation_joint",
    "y_rotation_joint",
    "z_rotation_joint",
]

KNOWN_URDF_FILENAMES = {
    "allegro": "allegro.urdf",
    "shadow": "shadowhand.urdf",
}

MANO_HAND_MODELS = {"mano"}

ROTATION_TO_VIEWER = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
TABLE_SIZE = (1.0, 1.0, 0.05)


def require_viewer_backend():
    if viewer_utils is not None and ViewerData is not None and ARCTICViewer is not object:
        return

    raise RuntimeError(
        "aitviewer is required for visualization, but it is not available in the current Python environment. "
        "Use the repository viewer environment, for example: "
        "conda run -n graspxl_viewer python scripts/visualizer_sharpa_table_top.py"
    )


def parse_collection_name(collection):
    """Parse '<hand_model>_<group>' collection names, e.g. 'sharpa_tabletop'."""
    parts = collection.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid collection '{collection}'. Expected '<hand_model>_<group>'."
        )

    hand_model = parts[0]
    group = "_".join(parts[1:])
    if not hand_model or not group:
        raise ValueError(
            f"Invalid collection '{collection}'. Expected '<hand_model>_<group>'."
        )
    return hand_model, group


def normalize_num_id(hand_model, num_id):
    """Accept both '0' and '<hand_model>_0(.npy)' num identifiers."""
    stem = op.splitext(str(num_id))[0]
    prefix = f"{hand_model}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def get_available_collections(data_root, hand_model_filter=None):
    if not op.isdir(data_root):
        return []

    collections = []
    for name in os.listdir(data_root):
        collection_dir = op.join(data_root, name)
        if not op.isdir(collection_dir):
            continue
        try:
            hand_model, _ = parse_collection_name(name)
        except ValueError:
            continue
        if hand_model_filter is not None and hand_model != hand_model_filter:
            continue
        if get_available_object_dicts(collection_dir, hand_model):
            collections.append(name)
    return sorted(collections)


def get_available_object_dicts(collection_dir, hand_model):
    if not op.isdir(collection_dir):
        return []

    object_dicts = []
    for name in os.listdir(collection_dir):
        object_dict_dir = op.join(collection_dir, name)
        if not op.isdir(object_dict_dir):
            continue
        if get_available_objects(object_dict_dir, hand_model):
            object_dicts.append(name)
    return sorted(object_dicts)


def get_available_objects(object_dict_dir, hand_model):
    if not op.isdir(object_dict_dir):
        return []

    objects = []
    for name in os.listdir(object_dict_dir):
        object_dir = op.join(object_dict_dir, name)
        if not op.isdir(object_dir):
            continue
        if glob.glob(op.join(object_dir, f"{hand_model}_*.npy")):
            objects.append(name)
    return sorted(objects)


def get_available_num_ids(object_dir, hand_model):
    npy_paths = glob.glob(op.join(object_dir, f"{hand_model}_*.npy"))
    prefix = f"{hand_model}_"
    return sorted(
        normalize_num_id(hand_model, op.basename(path))
        for path in npy_paths
        if op.splitext(op.basename(path))[0].startswith(prefix)
    )


def resolve_large_data_paths(
    data_root,
    collection,
    object_mesh_root,
    object_dict=None,
    object_name=None,
    num_id=None,
    hand_model_filter=None,
):
    """
    Resolve GraspXL large-format sequence and mesh paths.

    Returns:
        (collection, hand_model, group, object_dict, object_name, num_id, data_path, mesh_path)
        or None if any required path is missing.
    """
    if collection is None:
        available_collections = get_available_collections(data_root, hand_model_filter=hand_model_filter)
        if not available_collections:
            if hand_model_filter is None:
                logger.error(f"No large-format collections found in: {data_root}")
            else:
                logger.error(f"No '{hand_model_filter}' large-format collections found in: {data_root}")
            return None
        collection = random.choice(available_collections)
        logger.info(
            f"Randomly selected collection: {collection} "
            f"(from {len(available_collections)} available)"
        )

    try:
        hand_model, group = parse_collection_name(collection)
    except ValueError as exc:
        logger.error(str(exc))
        return None

    if hand_model_filter is not None and hand_model != hand_model_filter:
        logger.error(
            f"Collection '{collection}' is for hand model '{hand_model}', "
            f"but this entry point expects '{hand_model_filter}'."
        )
        return None

    collection_dir = op.join(data_root, collection)
    if not op.isdir(collection_dir):
        logger.error(f"Collection directory not found: {collection_dir}")
        return None

    available_object_dicts = get_available_object_dicts(collection_dir, hand_model)
    if not available_object_dicts:
        logger.error(f"No object_dict directories found in: {collection_dir}")
        return None

    if object_dict is None:
        object_dict = random.choice(available_object_dicts)
        logger.info(
            f"Randomly selected object_dict: {object_dict} "
            f"(from {available_object_dicts})"
        )
    elif object_dict not in available_object_dicts:
        logger.error(f"object_dict '{object_dict}' not found. Available: {available_object_dicts}")
        return None

    object_dict_dir = op.join(collection_dir, object_dict)
    available_objects = get_available_objects(object_dict_dir, hand_model)
    if not available_objects:
        logger.error(f"No sequence objects found in: {object_dict_dir}")
        return None

    if object_name is None:
        object_name = random.choice(available_objects)
        logger.info(
            f"Randomly selected object_name: {object_name} "
            f"(from {len(available_objects)} available)"
        )
    elif object_name not in available_objects:
        logger.error(f"Object '{object_name}' not found. Available examples: {available_objects[:10]}")
        return None

    object_dir = op.join(object_dict_dir, object_name)
    available_num_ids = get_available_num_ids(object_dir, hand_model)
    if not available_num_ids:
        logger.error(f"No '{hand_model}_*.npy' sequences found in: {object_dir}")
        return None

    if num_id is None:
        num_id = random.choice(available_num_ids)
        logger.info(f"Randomly selected num_id: {num_id} (from {available_num_ids})")
    else:
        num_id = normalize_num_id(hand_model, num_id)
        if num_id not in available_num_ids:
            logger.error(f"num_id '{num_id}' not found. Available: {available_num_ids}")
            return None

    data_path = op.join(object_dir, f"{hand_model}_{num_id}.npy")
    mesh_path = op.join(object_mesh_root, object_dict, object_name, f"{object_name}.obj")

    if not op.exists(data_path):
        logger.error(f"Sequence file not found: {data_path}")
        return None
    if not op.exists(mesh_path):
        logger.error(f"Mesh file not found: {mesh_path}")
        return None

    return collection, hand_model, group, object_dict, object_name, num_id, data_path, mesh_path


def resolve_urdf_path(hand_model, body_model_root):
    hand_root = op.join(body_model_root, hand_model)
    if not op.isdir(hand_root):
        logger.error(f"Hand model directory not found: {hand_root}")
        return None

    candidates = []
    known_name = KNOWN_URDF_FILENAMES.get(hand_model)
    if known_name is not None:
        candidates.append(op.join(hand_root, known_name))
    candidates.append(op.join(hand_root, f"{hand_model}.urdf"))
    candidates.extend(sorted(glob.glob(op.join(hand_root, "*.urdf"))))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if op.exists(candidate):
            return candidate

    logger.error(f"No URDF found in hand model directory: {hand_root}")
    return None


def prepare_urdf_for_urdfpy(urdf_path, hand_model_root, hand_model):
    """Rewrite mesh paths to absolute paths and fill missing urdfpy limit fields."""
    hand_model_root_abs = op.abspath(hand_model_root)
    urdf_dir_abs = op.abspath(op.dirname(urdf_path))
    root = ET.parse(urdf_path).getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            package_path = filename[len("package://"):]
            parts = package_path.split("/", 1)
            relative_path = parts[1] if len(parts) == 2 else ""
            mesh.set("filename", op.join(hand_model_root_abs, relative_path))
        elif not op.isabs(filename) and "://" not in filename:
            mesh.set("filename", op.join(urdf_dir_abs, filename))

    for limit in root.findall(".//limit"):
        limit.attrib.setdefault("velocity", "1.0")
        limit.attrib.setdefault("effort", "1.0")

    temp_path = op.join(tempfile.gettempdir(), f"graspxl_{hand_model}_urdfpy.urdf")
    ET.ElementTree(root).write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def axisangle2mat(rot_vecs):
    import torch

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat.numpy()


def get_configurable_joint_names(robot):
    return [joint.name for joint in robot.joints if joint.joint_type != "fixed"]


def make_joint_config(joint_names, pose_values):
    pose_values = np.asarray(pose_values).reshape(-1)
    cfg = {joint_name: 0.0 for joint_name in joint_names}
    has_virtual_prefix = joint_names[: len(VIRTUAL_JOINTS)] == VIRTUAL_JOINTS

    if has_virtual_prefix and pose_values.shape[0] == len(joint_names):
        for joint_name, value in zip(joint_names[len(VIRTUAL_JOINTS):], pose_values[len(VIRTUAL_JOINTS):]):
            cfg[joint_name] = float(value)
        return cfg

    if not has_virtual_prefix and pose_values.shape[0] == len(joint_names):
        for joint_name, value in zip(joint_names, pose_values):
            cfg[joint_name] = float(value)
        return cfg

    if has_virtual_prefix and pose_values.shape[0] == len(joint_names) - len(VIRTUAL_JOINTS):
        for joint_name, value in zip(joint_names[len(VIRTUAL_JOINTS):], pose_values):
            cfg[joint_name] = float(value)
        return cfg

    for joint_name, value in zip(joint_names, pose_values):
        cfg[joint_name] = float(value)
    return cfg


def apply_object_tabletop_alignment(v3d_o, hand_vertices):
    import torch

    v3d_o_tensor = torch.from_numpy(v3d_o)
    centers_xy = v3d_o_tensor.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
    centers_xy[:, :, 2] = 0
    z_min = v3d_o_tensor[:, :, 2].min()

    v3d_o -= centers_xy.numpy()
    v3d_o[:, :, 2] -= z_min.numpy()
    for verts in hand_vertices:
        verts -= centers_xy.numpy()
        verts[:, :, 2] -= z_min.numpy()
    return centers_xy.numpy(), float(z_min.numpy())


def construct_urdf_meshes_per_object(obj_key, raw_data, mesh_path, robot, joint_names, hand_model):
    import torch
    import trimesh

    trans_r = torch.from_numpy(raw_data["right_hand"]["trans"])
    rot_r = torch.from_numpy(raw_data["right_hand"]["rot"])
    pose_r = torch.from_numpy(raw_data["right_hand"]["pose"])

    trans_o = torch.from_numpy(raw_data[obj_key]["trans"])
    rot_o = torch.from_numpy(raw_data[obj_key]["rot"])

    num_frames = trans_r.shape[0]
    hand_rot_mats = axisangle2mat(-rot_r)

    if pose_r.shape[1] != len(joint_names):
        has_virtual_prefix = joint_names[: len(VIRTUAL_JOINTS)] == VIRTUAL_JOINTS
        accepts_without_virtual = has_virtual_prefix and pose_r.shape[1] == len(joint_names) - len(VIRTUAL_JOINTS)
        if not accepts_without_virtual:
            logger.warning(
                f"Pose width ({pose_r.shape[1]}) does not match configurable joints ({len(joint_names)}). "
                "Missing joints will be zero-filled and extra pose values ignored."
            )

    fk0 = robot.visual_trimesh_fk(cfg={joint_name: 0.0 for joint_name in joint_names})
    visual_meshes0 = list(fk0.keys())
    hand_faces = [mesh.faces for mesh in visual_meshes0]
    hand_vertices = [
        np.zeros((num_frames, mesh.vertices.shape[0], 3))
        for mesh in visual_meshes0
    ]

    for frame_idx in range(num_frames):
        cfg = make_joint_config(joint_names, pose_r[frame_idx].numpy())
        fk = robot.visual_trimesh_fk(cfg=cfg)
        visual_meshes = list(fk.keys())
        for mesh_idx, mesh in enumerate(visual_meshes):
            pose = fk[mesh]
            temp_verts = np.ones((mesh.vertices.shape[0], 4))
            temp_verts[:, :3] = mesh.vertices
            temp_verts = np.matmul(temp_verts, pose.T)[:, :3]
            hand_vertices[mesh_idx][frame_idx] = (
                np.matmul(temp_verts, hand_rot_mats[frame_idx])
                + np.tile(trans_r.numpy()[frame_idx, :].reshape(1, 3), (temp_verts.shape[0], 1))
            )

    mesh = trimesh.load(mesh_path, process=False)
    object_v = mesh.vertices
    object_f = mesh.faces

    object_rot_mats = axisangle2mat(-rot_o)
    v3d_o = np.zeros((trans_o.shape[0], object_v.shape[0], 3))
    for frame_idx in range(trans_o.shape[0]):
        v3d_o[frame_idx] = (
            np.matmul(object_v, object_rot_mats[frame_idx])
            + np.tile(trans_o.numpy()[frame_idx, :].reshape(1, 3), (object_v.shape[0], 1))
        )

    table_alignment = apply_object_tabletop_alignment(v3d_o, hand_vertices)

    mesh_specs = {}
    for mesh_idx, (v3d, f3d) in enumerate(zip(hand_vertices, hand_faces), start=1):
        mesh_specs[f"right{mesh_idx}"] = {
            "v3d": v3d,
            "f3d": f3d,
            "vc": None,
            "name": "right",
            "color": "mixer",
            "texture": None,
            "uv": None,
        }

    mesh_specs["object"] = {
        "v3d": v3d_o,
        "f3d": object_f,
        "vc": None,
        "name": "object",
        "color": "light-blue",
        "texture": None,
        "uv": None,
    }

    meshes = viewer_utils.construct_viewer_meshes(mesh_specs, draw_edges=False, flat_shading=True)
    for mesh in meshes.values():
        mesh.rotation = ROTATION_TO_VIEWER
    return meshes, num_frames, table_alignment


def construct_mano_meshes_per_object(obj_key, raw_data, mesh_path):
    import torch
    import trimesh
    from aitviewer.renderables.arrows import Arrows
    from common.body_models import seal_mano_mesh
    from smplx import MANO

    pose_r = torch.from_numpy(np.concatenate((raw_data["right_hand"]["rot"], raw_data["right_hand"]["pose"]), axis=1))
    trans_r = torch.from_numpy(raw_data["right_hand"]["trans"])

    trans_o = torch.from_numpy(raw_data[obj_key]["trans"])
    rot_o = torch.from_numpy(raw_data[obj_key]["rot"])

    mano_layer = MANO(
        "./data/body_models/mano",
        create_transl=False,
        use_pca=False,
        flat_hand_mean=True,
        is_rhand=True,
    ).to("cpu")

    pose_r = pose_r.reshape(-1, 48)
    cam_r = trans_r.view(-1, 1, 3)
    num_frames = trans_o.shape[0]

    out_r = mano_layer(
        global_orient=pose_r[:, :3].reshape(-1, 3),
        hand_pose=pose_r[:, 3:].reshape(-1, 45),
        betas=torch.zeros((num_frames, 10)).view(-1, 10),
    )

    wrist_bias = torch.tensor([0.09566994, 0.00638343, 0.0061863]).view(1, 1, 3)
    v3d_r = out_r.vertices - wrist_bias + cam_r

    f3d_r = torch.LongTensor(mano_layer.faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    object_v = mesh.vertices
    f3d_o = mesh.faces

    object_rot_mats = axisangle2mat(-rot_o)
    v3d_o = np.zeros((trans_o.shape[0], object_v.shape[0], 3))
    for frame_idx in range(trans_o.shape[0]):
        v3d_o[frame_idx] = (
            np.matmul(object_v, object_rot_mats[frame_idx])
            + np.tile(trans_o.numpy()[frame_idx, :].reshape(1, 3), (object_v.shape[0], 1))
        )

    hand_vertices = [v3d_r.numpy()]
    table_alignment = apply_object_tabletop_alignment(v3d_o, hand_vertices)
    v3d_r = hand_vertices[0]

    arrow = None
    if "points" in raw_data["right_hand"]:
        line_origin = torch.from_numpy(raw_data["right_hand"]["points"])[:, 1, :, :].numpy()
        line_tips = torch.from_numpy(raw_data["right_hand"]["points"])[:, 0, :, :].numpy()
        centers_xy, z_min = table_alignment
        line_origin -= centers_xy
        line_tips -= centers_xy
        line_origin[:, :, 2] -= z_min
        line_tips[:, :, 2] -= z_min
        rotation = np.array(ROTATION_TO_VIEWER).reshape(3, 3)
        line_origin = np.matmul(line_origin, rotation.T)
        line_tips = np.matmul(line_tips, rotation.T)
        arrow = Arrows(line_origin, line_tips, r_base=0.002, r_head=0.004, color=(0.969, 0.106, 0.059, 1.0))

    mesh_specs = {
        "right": {
            "v3d": v3d_r,
            "f3d": f3d_r.numpy(),
            "vc": None,
            "name": "right",
            "color": "white",
            "texture": None,
            "uv": None,
        },
        "object": {
            "v3d": v3d_o,
            "f3d": f3d_o,
            "vc": None,
            "name": "object",
            "color": "light-blue",
            "texture": None,
            "uv": None,
        },
    }

    meshes = viewer_utils.construct_viewer_meshes(mesh_specs, draw_edges=False, flat_shading=True)
    for mesh_obj in meshes.values():
        mesh_obj.rotation = ROTATION_TO_VIEWER
    return meshes, num_frames, arrow, table_alignment


def make_table_mesh(num_frames, center_xy=(0.0, 0.0), top_z=0.0, size=TABLE_SIZE):
    import trimesh

    size_x, size_y, size_z = size
    center_x, center_y = center_xy
    center_z = top_z - size_z / 2.0
    mesh = trimesh.creation.box(extents=(size_x, size_y, size_z))
    mesh.apply_translation((center_x, center_y, center_z))
    return np.repeat(mesh.vertices[None, :, :], num_frames, axis=0), mesh.faces


def add_table_to_batch(batch):
    meshes_all, data = batch
    if "table_alignment" in data:
        centers_xy, z_min = data["table_alignment"]
        center_xy = -np.asarray(centers_xy).reshape(-1, 3)[0, :2]
        top_z = -float(z_min)
    else:
        object_vertices = []
        for key, mesh in meshes_all.items():
            if key == "table" or key.startswith("right"):
                continue
            object_vertices.append(np.asarray(mesh.vertices))

        if object_vertices:
            all_object_vertices = np.concatenate(
                [vertices.reshape(-1, 3) for vertices in object_vertices],
                axis=0,
            )
            min_xy = all_object_vertices[:, :2].min(axis=0)
            max_xy = all_object_vertices[:, :2].max(axis=0)
            center_xy = (min_xy + max_xy) / 2.0
            top_z = float(all_object_vertices[:, 2].min())
        else:
            center_xy = np.array([0.0, 0.0])
            top_z = 0.0

    table_v, table_f = make_table_mesh(data.num_frames, center_xy=center_xy, top_z=top_z)
    table_mesh = viewer_utils.construct_viewer_meshes(
        {
            "table": {
                "v3d": table_v,
                "f3d": table_f,
                "vc": None,
                "name": "table",
                "color": "box",
                "texture": None,
                "uv": None,
            }
        },
        draw_edges=False,
        flat_shading=True,
    )["table"]
    table_mesh.rotation = ROTATION_TO_VIEWER
    meshes_all["table"] = table_mesh
    return meshes_all, data


class GraspXLLargeViewer(ARCTICViewer):
    def load_data(self, data_path, mesh_path, hand_model, body_model_root):
        require_viewer_backend()

        from common.xdict import xdict
        from urdfpy import URDF

        logger.info("Creating meshes")
        graspxl_data = np.load(data_path, allow_pickle=True).item()

        object_keys = list(graspxl_data.keys())
        object_keys.remove("right_hand")

        urdf_path = resolve_urdf_path(hand_model, body_model_root)
        if urdf_path is None:
            raise FileNotFoundError(f"Could not resolve URDF for hand model '{hand_model}'.")
        hand_model_root = op.join(body_model_root, hand_model)
        prepared_urdf_path = prepare_urdf_for_urdfpy(urdf_path, hand_model_root, hand_model)

        robot = URDF.load(prepared_urdf_path)
        joint_names = get_configurable_joint_names(robot)
        logger.info(f"Using hand model URDF: {urdf_path}")
        logger.info(f"Using {len(joint_names)} configurable joints")

        meshes_all = xdict()
        num_frames = None
        table_alignment = None
        for obj_key in object_keys:
            meshes, num_frames, current_alignment = construct_urdf_meshes_per_object(
                obj_key=obj_key,
                raw_data=graspxl_data,
                mesh_path=mesh_path,
                robot=robot,
                joint_names=joint_names,
                hand_model=hand_model,
            )
            if table_alignment is None:
                table_alignment = current_alignment
            if len(meshes_all) > 0:
                for key in list(meshes.keys()):
                    if key.startswith("right"):
                        meshes.pop(key)

            obj_mesh = meshes.pop("object")
            obj_mesh.name = obj_key
            meshes[obj_key] = obj_mesh
            meshes_all.merge(meshes)

        focal = 1000.0
        rows = 224
        cols = 224
        k_matrix = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
        rt_matrix = np.zeros((num_frames, 3, 4))
        rt_matrix[:, :3, :3] = np.eye(3)
        rt_matrix[:, 1:3, :3] *= -1.0

        data = ViewerData(Rt=rt_matrix, K=k_matrix, cols=cols, rows=rows, imgnames=None)
        data["table_alignment"] = table_alignment
        batch = meshes_all, data
        self.check_format(batch)
        logger.info("Done")
        return batch


class GraspXLManoTableViewer(ARCTICViewer):
    def load_data(self, data_path, mesh_path):
        require_viewer_backend()

        from common.xdict import xdict

        logger.info("Creating MANO meshes")
        graspxl_data = np.load(data_path, allow_pickle=True).item()

        object_keys = list(graspxl_data.keys())
        object_keys.remove("right_hand")

        meshes_all = xdict()
        num_frames = None
        arrow = None
        table_alignment = None
        for obj_key in object_keys:
            meshes, num_frames, arrow, current_alignment = construct_mano_meshes_per_object(
                obj_key=obj_key,
                raw_data=graspxl_data,
                mesh_path=mesh_path,
            )
            if table_alignment is None:
                table_alignment = current_alignment
            if len(meshes_all) > 0 and "right" in meshes:
                meshes.pop("right")

            obj_mesh = meshes.pop("object")
            obj_mesh.name = obj_key
            meshes[obj_key] = obj_mesh
            meshes_all.merge(meshes)

        focal = 1000.0
        rows = 224
        cols = 224
        k_matrix = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
        rt_matrix = np.zeros((num_frames, 3, 4))
        rt_matrix[:, :3, :3] = np.eye(3)
        rt_matrix[:, 1:3, :3] *= -1.0

        data = ViewerData(Rt=rt_matrix, K=k_matrix, cols=cols, rows=rows, imgnames=None)
        data["table_alignment"] = table_alignment
        batch = meshes_all, data
        self.check_format(batch)
        logger.info("Done")
        return batch, arrow


def parse_args(description=None, collection_example="sharpa_tabletop", hand_model_example="sharpa"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exp_folder", type=str, default="")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Collection folder, e.g. {collection_example}. Random if not specified.",
    )
    parser.add_argument("--object_dict", type=str, default=None, help="Object dictionary folder, e.g. large. Random if not specified.")
    parser.add_argument("--object_mesh_root", type=str, default=DEFAULT_OBJECT_MESH_ROOT)
    parser.add_argument("--body_model_root", type=str, default=DEFAULT_BODY_MODEL_ROOT)
    parser.add_argument("--object_name", type=str, default=None, help="Object folder/name. Random if not specified.")
    parser.add_argument(
        "--num_id",
        type=str,
        default=None,
        help=f"Sequence number, e.g. 0, or full stem, e.g. {hand_model_example}_0. Random if not specified.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_table", action="store_true", help="Disable the 1m x 1m x 0.05m table under the object.")
    return parser.parse_args()


def main(script_name="visualizer_sharpa_table_top.py", hand_model_filter="sharpa", collection_example="sharpa_tabletop"):
    args = parse_args(
        description=f"Visualize {hand_model_filter} GraspXL large-format sequences with a 1m table top.",
        collection_example=collection_example,
        hand_model_example=hand_model_filter,
    )

    result = resolve_large_data_paths(
        data_root=args.data_root,
        collection=args.collection,
        object_mesh_root=args.object_mesh_root,
        object_dict=args.object_dict,
        object_name=args.object_name,
        num_id=args.num_id,
        hand_model_filter=hand_model_filter,
    )
    if result is None:
        logger.error("Failed to resolve GraspXL large-format paths. Check your parameters.")
        return

    collection, hand_model, group, object_dict, object_name, num_id, data_path, mesh_path = result

    try:
        require_viewer_backend()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Rendering {collection}/{object_dict}/{object_name}/{hand_model}_{num_id}.npy")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Mesh path: {mesh_path}")

    render_size = (2560, 1440) if args.headless else (8024, 4024)
    arrow = None
    if hand_model in MANO_HAND_MODELS:
        viewer = GraspXLManoTableViewer(
            interactive=not args.headless,
            size=render_size,
            render_types=["rgb"],
        )
        batch, arrow = viewer.load_data(data_path, mesh_path)
    else:
        viewer = GraspXLLargeViewer(
            interactive=not args.headless,
            size=render_size,
            render_types=["rgb"],
        )
        batch = viewer.load_data(
            data_path=data_path,
            mesh_path=mesh_path,
            hand_model=hand_model,
            body_model_root=args.body_model_root,
        )

    if not args.no_table:
        batch = add_table_to_batch(batch)

    if args.headless:
        out_folder = op.join(
            args.exp_folder,
            "render",
            "GraspXL",
            collection,
            object_dict,
            object_name,
            f"{hand_model}_{num_id}",
        )
    else:
        out_folder = ""

    video_name = f"{object_name}_{hand_model}_{num_id}"
    viewer.render_seq(batch, out_folder=out_folder, object_name=video_name, arrow=arrow)

    cmd_parts = [f"python scripts/{script_name}"]
    cmd_parts.append(f"--collection {collection}")
    cmd_parts.append(f"--object_dict {object_dict}")
    cmd_parts.append(f"--object_name {object_name}")
    cmd_parts.append(f"--num_id {num_id}")
    if args.data_root != DEFAULT_DATA_ROOT:
        cmd_parts.append(f"--data_root {args.data_root}")
    if args.object_mesh_root != DEFAULT_OBJECT_MESH_ROOT:
        cmd_parts.append(f"--object_mesh_root {args.object_mesh_root}")
    if args.body_model_root != DEFAULT_BODY_MODEL_ROOT:
        cmd_parts.append(f"--body_model_root {args.body_model_root}")
    if args.headless:
        cmd_parts.append("--headless")
    if args.no_table:
        cmd_parts.append("--no_table")
    if args.exp_folder:
        cmd_parts.append(f"--exp_folder {args.exp_folder}")

    reproducible_cmd = " ".join(cmd_parts)
    logger.info(f"Reproducible command:\n{reproducible_cmd}")

    if args.headless and out_folder:
        config = {
            "timestamp": datetime.now().isoformat(),
            "command": reproducible_cmd,
            "parameters": {
                "data_root": args.data_root,
                "collection": collection,
                "hand_model": hand_model,
                "group": group,
                "object_dict": object_dict,
                "object_mesh_root": args.object_mesh_root,
                "body_model_root": args.body_model_root,
            },
            "sequence": {
                "object_name": object_name,
                "num_id": num_id,
                "data_path": data_path,
                "mesh_path": mesh_path,
            },
        }
        os.makedirs(out_folder, exist_ok=True)
        config_path = op.join(out_folder, "render_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        logger.info(f"Saved render config to: {config_path}")


if __name__ == "__main__":
    main()
