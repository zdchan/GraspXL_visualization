import os
import os.path as op
import re
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.cm as cm
import numpy as np
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from aitviewer.utils.so3 import aa2rot_numpy
from loguru import logger
from PIL import Image
from tqdm import tqdm
from aitviewer.utils import path

from aitviewer.headless import HeadlessRenderer
from aitviewer.viewer import Viewer
from easydict import EasyDict as edict
from aitviewer.scene.camera import PinholeCamera

cmap = cm.get_cmap("plasma")
materials = {
    "none": None,
    "white": Material(color=(1.0, 1.0, 1.0, 1.0), ambient=0.2),
    "red": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.3),
    "blue": Material(color=(0.0, 0.0, 1.0, 1.0), ambient=0.3),
    "green": Material(color=(1.0, 0.0, 0.0, 1.0), ambient=0.3),
    "cyan": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.3),
    "light-blue": Material(color=(0.588, 0.5647, 0.9725, 1.0), ambient=0.3),
    "cyan-light": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.3),
    "dark-light": Material(color=(59.0/255.0, 39.0/255.0, 39.0/255.0, 1.0), ambient=0.3),
    "wood": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.3),
    "box": Material(color=(0.835, 0.651, 0.49, 255/255.0), ambient=0.3),
    "ketchup": Material(color=(0.969, 0.106, 0.059, 255.0/255.0), ambient=0.3),
    "mixer": Material(color=(135.0/255.0, 135.0/255.0, 166.0/255.0, 255.0/255.0), ambient=0.3),
    "other": Material(color=(115.0/255.0, 150.0/255.0, 129.0/255.0, 255.0/255.0), ambient=0.3),
    "rice": Material(color=(0.922, 0.922, 0.102, 1.0), ambient=0.3),
}


class ViewerData(edict):
    """
    Interface to standardize viewer data.
    """

    def __init__(self, Rt, K, cols, rows, imgnames=None):
        self.imgnames = imgnames
        self.Rt = Rt
        self.K = K
        self.num_frames = Rt.shape[0]
        self.cols = cols
        self.rows = rows
        self.validate_format()

    def validate_format(self):
        assert len(self.Rt.shape) == 3
        assert self.Rt.shape[0] == self.num_frames
        assert self.Rt.shape[1] == 3
        assert self.Rt.shape[2] == 4

        assert len(self.K.shape) == 2
        assert self.K.shape[0] == 3
        assert self.K.shape[1] == 3
        if self.imgnames is not None:
            assert self.num_frames == len(self.imgnames)
            assert self.num_frames > 0
            im_p = self.imgnames[0]
            assert op.exists(im_p), f"Image path {im_p} does not exist"


class ARCTICViewer:
    def __init__(
            self,
            render_types=["rgb", "depth", "mask"],
            interactive=True,
            size=(2024, 2024),
    ):
        if not interactive:
            v = HeadlessRenderer(size=size)
        else:
            v = Viewer(size=size)

        self.v = v
        self.interactive = interactive
        # self.layers = layers
        self.render_types = render_types

    def view_interactive(self):
        # self.v.scene.camera.load_cam()
        self.v.run()

    def view_fn_headless(self, num_iter, out_folder, object_name):
        v = self.v

        v._init_scene()
        # v.scene.camera.load_cam()

        logger.info("Rendering to video")
        vid_p = op.join(out_folder, f"ours/{object_name}.mp4")
        v.save_video(video_dir=vid_p)

        # pbar = tqdm(range(num_iter))
        # # out_rgb_folder = op.join(out_folder, "images", f"/{object_name}/")
        # # if not os.path.exists(out_rgb_folder):
        # #     os.makedirs(out_rgb_folder)
        # for fidx in pbar:
        #     out_rgb = op.join(out_folder, "images", f"rgb/{object_name}/{fidx:04d}.png")
        #     out_mask = op.join(out_folder, "images", f"mask/{fidx:04d}.png")
        #     out_depth = op.join(out_folder, "images", f"depth/{fidx:04d}.npy")
        #
        #     # render RGB, depth, segmentation masks
        #     if "rgb" in self.render_types:
        #         v.export_frame(out_rgb, transparent_background=False)
        #     if "depth" in self.render_types:
        #         os.makedirs(op.dirname(out_depth), exist_ok=True)
        #         render_depth(v, out_depth)
        #     if "mask" in self.render_types:
        #         os.makedirs(op.dirname(out_mask), exist_ok=True)
        #         render_mask(v, out_mask)
        #     v.scene.next_frame()
        # logger.info(f"Exported to {out_folder}")

    @abstractmethod
    def load_data(self):
        pass

    def check_format(self, batch):
        meshes_all, data = batch
        assert isinstance(meshes_all, dict)
        assert len(meshes_all) > 0
        for mesh in meshes_all.values():
            assert isinstance(mesh, Meshes)
        assert isinstance(data, ViewerData)

    def render_seq(self, batch, out_folder="./render_out", object_name="object", arrow=None):
        meshes_all, data = batch
        self.setup_viewer(data)
        for mesh in meshes_all.values():
            self.v.scene.add(mesh)
        if arrow is not None:
            self.v.scene.add(arrow)
        if self.interactive:
            self.view_interactive()
        else:
            num_iter = data["num_frames"]
            self.view_fn_headless(num_iter, out_folder, object_name)

    def setup_viewer(self, data):
        v = self.v
        fps = 30
        if "imgnames" in data:
            setup_billboard(data, v)

        # camera.show_path()
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        v.scene.floor.position[1] = -3
        v.auto_set_camera_target = False

        # d = 6 # Distance from the object at start and end.
        # r = 0  # Radius of the circle around the object.
        # h = 5.0  # Height of the circle.
        # first = path.line(start=(r, 0, 0), end=(r, h, d), num=140)
        # # circle = path.circle(
        # #     center=(0, h, 0),
        # #     radius=r,
        # #     num=int(314 * 2 * r / d),
        # #     start_angle=360,
        # #     end_angle=0,
        # # )
        # # second = path.line(start=(r, h, 0), end=(r, h, -d), num=100)
        # positions = np.vstack((first))
        # # positions = first
        #
        # # direction = np.array([1, 1, 1])/3
        # # positions = np
        #
        #
        # targets = np.array([0, 0, 0])
        # camera = PinholeCamera(positions, targets, v.window_size[0], v.window_size[1], viewer=v)
        #
        #
        #
        # # Add the camera and the SMPL sequence to the scene.
        # v.scene.add(camera)
        #
        # # Set the camera as the current viewer camera.
        # v.set_temp_camera(camera)
        # v.scene.camera.position = np.array((0.0, 0.0, 0))
        self.v = v


def dist2vc_segm(contact_t, num_parts):
    ccmap = matplotlib.cm.get_cmap("prism")
    contact_t_norm = contact_t / num_parts
    vc = ccmap(contact_t_norm)
    return vc


def dist2vc(dist_ro, dist_lo, dist_o, _cmap, tf_fn=None):
    if tf_fn is not None:
        exp_map = tf_fn
    else:
        exp_map = small_exp_map
    dist_ro = exp_map(dist_ro)
    dist_lo = exp_map(dist_lo)
    dist_o = exp_map(dist_o)

    vc_ro = _cmap(dist_ro)
    vc_lo = _cmap(dist_lo)
    vc_o = _cmap(dist_o)
    return vc_ro, vc_lo, vc_o


def dist2vc_bin(_dist):
    dist = np.copy(_dist)
    dist = (dist < 0.01).astype(np.float32)
    vc = cmap(dist)
    return vc


def small_exp_map(_dist):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-20.0 * dist)
    return dist


def dist2vc_cont_exp(_dist, _cmap):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-10.0 * dist)
    dist = dist / dist.max()
    # dist += 0.1
    vc = _cmap(dist)
    return vc


def dist2vc_cont(_dist, _cmap):
    dist = np.copy(_dist)
    dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    vc = _cmap(dist)
    return vc


def construct_viewer_meshes(data, draw_edges=False, flat_shading=True):
    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    meshes = {}
    for key, val in data.items():
        if "object" in key:
            flat_shading = True
        else:
            flat_shading = False
        v3d = val["v3d"]
        meshes[key] = Meshes(
            v3d,
            val["f3d"],
            vertex_colors=val["vc"],
            name=val["name"],
            flat_shading=flat_shading,
            draw_edges=draw_edges,
            material=materials[val["color"]],
            rotation=rotation_flip,
            uv_coords=val["uv"],
            path_to_texture=val["texture"],
        )
    return meshes


def setup_viewer(
        v, shared_folder_p, video, images_path, data, flag, seq_name, side_angle
):
    fps = 10
    cols, rows = 224, 224
    focal = 1000.0

    # setup image paths
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # setup billboard
    images_path = op.join(shared_folder_p, "images")
    images_paths = [
        os.path.join(images_path, f)
        for f in sorted(os.listdir(images_path), key=sort_key)
    ]
    assert len(images_paths) > 0

    cam_t = data[f"{flag}.object.cam_t"]
    num_frames = min(cam_t.shape[0], len(images_paths))
    cam_t = cam_t[:num_frames]
    # setup camera
    K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    # camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if side_angle is None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    # v.scene.add(camera)
    v.run_animations = True  # autoplay
    v.playback_fps = fps
    v.scene.fps = fps
    v.scene.origin.enabled = False
    v.scene.floor.enabled = False
    v.auto_set_floor = False
    v.scene.floor.position[1] = -3
    # v.set_temp_camera(camera)
    # v.scene.camera.position = np.array((0.0, 0.0, 0))
    return v


def render_depth(v, depth_p):
    depth = np.array(v.get_depth()).astype(np.float16)
    np.save(depth_p, depth)


def render_mask(v, mask_p):
    mask = np.array(v.get_mask()).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(mask_p)


def setup_billboard(data, v):
    images_paths = data.imgnames
    K = data.K
    Rt = data.Rt
    rows = data.rows
    cols = data.cols
    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if images_paths is not None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    # v.scene.add(camera)
    v.scene.camera.load_cam()
    # v.set_temp_camera(camera)