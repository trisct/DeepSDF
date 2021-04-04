import os
import argparse
import numpy as np
import pyrender
import open3d as o3d
import trimesh

def get_normal_line_set(points, normals, length=.005):
    normals_end = points + normals * length
    points_normals = np.concatenate([points, normals_end], axis=0)
    
    lines_start_idx = np.linspace(0, len(points)-1, len(points)).astype(np.int64)
    print(len(points))
    lines_end_idx = lines_start_idx + len(points)
    
    print(lines_start_idx)
    print(lines_end_idx)
    lines_normals = np.stack([lines_start_idx, lines_end_idx], axis=1)
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_normals),
        lines=o3d.utility.Vector2iVector(lines_normals),
    )
    return line_set

def normal_sample_handler(filename, return_pts_nml=True):
    """
    filename: path without '_points.obj' or '_normals.obj' suffixes
    """
    points = trimesh.load(filename + '_points.obj')
    normals = trimesh.load(filename + '_normals.obj')
    
    points = np.array(points.vertices).copy()
    normals = np.array(normals.vertices).copy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd, points, normals
    
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', nargs='+')

args = parser.parse_args()

if __name__ == '__main__':
    items = args.input

    print(items)
    pcd, points, normals = normal_sample_handler(items[0])
    nml = get_normal_line_set(points, normals)

    o3d.visualization.draw_geometries([pcd, nml])
    o3d.io.write_point_cloud(items[0]+'.ply', pcd)
