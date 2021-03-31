import os
import argparse
import numpy as np
import pyrender
import open3d as o3d

class OnscreenRenderer(object):
    def __init__(self):
        pass
    def __call__(self, scene):
        pyrender.Viewer(scene, use_raymond_lighting=True)

def setup_scene_and_renderer(mode):
    """
    This function returns a tuple (scene, renderer) with default settings.

    mode: str, 'onscreen' or 'offscreen'
    """
    
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[1., 1., 1.])

    if mode == 'onscreen':
        renderer = OnscreenRenderer()
    elif mode == 'offscreen':
        pers_cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
        node_cam = pyrender.Node(camera=pers_cam, matrix=np.eye(4))
        camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0]])
        scene.add_node(node_cam)    
        scene.set_pose(node_cam, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer
    else:
        print(f'Unsupported mode {mode}')
        return

    return scece, renderer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', nargs='+')

args = parser.parse_args()

def sdf_sample_handler(filename, mode='both'):
    """
    mode: str, 'both', 'pos' or 'neg'
    """
    data = np.load(filename)
    
    if mode == 'pos' or mode == 'neg':
        points = data[mode][:, :3]
        colors = data[mode][:, 3:]
        colors = np.repeat(colors, 3, axis=1)
    elif mode =='both':
        points = np.concatenate([data['pos'][:, :3], data['neg'][:, :3]], axis=0)
        colors = np.concatenate([data['pos'][:, 3:], data['neg'][:, 3:]], axis=0)
        colors = np.repeat(colors, 3, axis=1)
    else:
        print(f'Unsupported mode: {mode}')
        return
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return points, colors

def surface_sample_handler(filename):
    """
    filename: str, should be obj stored in NormalSamples.
    """
    points = trimesh.load('')
    data = np.load(filename)
    
    if mode == 'pos' or mode == 'neg':
        points = data[mode][:, :3]
        colors = data[mode][:, 3:]
        colors = np.repeat(colors, 3, axis=1)
    elif mode =='both':
        points = np.concatenate([data['pos'][:, :3], data['neg'][:, :3]], axis=0)
        colors = np.concatenate([data['pos'][:, 3:], data['neg'][:, 3:]], axis=0)
        colors = np.repeat(colors, 3, axis=1)
    else:
        print(f'Unsupported mode: {mode}')
        return
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return points, colors

if __name__ == '__main__':
    items = args.input

    print(items)
    z = sdf_sample_handler(items[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(z[0])
    pcd.colors = o3d.utility.Vector3dVector(z[1])

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(items[0][:-4]+'.ply', pcd)
