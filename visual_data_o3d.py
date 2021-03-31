import os
import argparse
import numpy as np
import pyrender
import open3d as o3d

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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', nargs='+')

args = parser.parse_args()

if __name__ == '__main__':
    items = args.input

    print(items)
    z = sdf_sample_handler(items[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(z[0])
    pcd.colors = o3d.utility.Vector3dVector(z[1])

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(items[0][:-4]+'.ply', pcd)
