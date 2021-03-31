import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
import torch
import torchvision
import glob

def render_one(mesh_list, steps, save_name, save_path, resolution, need_video=False):
    """
    mesh: pyrender.mesh.Mesh
        A pyrender.mesh.Mesh object
    steps: int
        number of steps in one horizontal revolution
    save_path: str
        path to save color and depth image (saved as numpy arrays).
    mode: str, either 'light' or 'albedo'
        if 'light', then render with light objects
        if 'albedo', then render with only ambient lights
    resolution: tuple of 2: (res_h, res_w)

    ----
    file saving:
        This files save the color image, the depth image, the camera pose and the camera projection matrix
        
        color image: saved as
            [save_path]/[save_name]/[save_name]_[rotate_deg]_color.npy
        depth image: saved as
            [save_path]/[save_name]/[save_name]_[rotate_deg]_depth.npy
        camera pose: saved as
            [save_path]/[save_name]/[save_name]_[rotate_deg]_campose.npy
        projection matrix: saved as
            [save_path]/[save_name]/[save_name]_projection.npy
    """
    print(f'Starting to render one, which will be saved to {os.path.join(save_path, save_name)}.')
    if not os.path.exists(os.path.join(save_path, save_name)):
        os.system(f'mkdir -p {os.path.join(save_path, save_name)}')

    # resolution
    res_h, res_w = resolution

    # creating nodes
    # mesh
    node_mesh_list = []
    for mesh in mesh_list:
        #mesh = pyrender.Mesh.from_trimesh(mesh)
        node_mesh_list.append( pyrender.Node(mesh=mesh, matrix=np.eye(4)) )

    # directional light
    dir_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    node_light = pyrender.Node(light=dir_light, matrix=np.eye(4))

    # perspective cameras
    pers_cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1)
    node_cam = pyrender.Node(camera=pers_cam, matrix=np.eye(4))

    # scene
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[1., 1., 1.])
    for node_mesh in node_mesh_list:
        scene.add_node(node_mesh)
    scene.add_node(node_light)
    scene.add_node(node_cam)
    

    offscr_renderer = pyrender.OffscreenRenderer(viewport_width=res_h, viewport_height=res_w, point_size=3.)
    
    
    # for outputting video
    if need_video:
        color_video = torch.zeros(steps, res_h, res_w, 3, dtype=torch.uint8)
        depth_video = torch.zeros(steps, res_h, res_w, 3, dtype=torch.uint8)
        albedo_video = torch.zeros(steps, res_h, res_w, 3, dtype=torch.uint8)
    
    deg_interval = 720 / steps
    
    for i, angle_i in enumerate(range(steps)):
        print(f'Showing angle {angle_i}')
        angle_deg = int(deg_interval * angle_i)
        angle_rad = angle_deg * math.pi / 180

        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        camera_pose = np.array([
            [  c,  0.0,   s, 2*s],
            [0.0,  -1.0, 0.0, 0.0],
            [  s,  0.0,   -c, -2*c],
            [0.0,  0.0, 0.0, 1.0]])

        pitch_angle_rad = 30 * math.pi / 180
        s_pitch = math.sin(pitch_angle_rad)
        c_pitch = math.cos(pitch_angle_rad)


        # rendering
        scene.set_pose(node_cam, pose=camera_pose)
        color, depth = offscr_renderer.render(scene)                
        scene.remove_node(node_light)
        albedo, _ = offscr_renderer.render(scene)
        scene.add_node(node_light)
		
        #plt.imshow(color)
        #plt.show()
		
        # making video
        if need_video:
            color_video[i] = torch.from_numpy(color.copy())
        
            depth_pt = torch.from_numpy(depth.copy())
            depth_scaled = (depth_pt - depth_pt[depth_pt !=0].min()) / (depth_pt[depth_pt != 0].max() - depth_pt[depth_pt != 0].min()) * 255
            depth_scaled = torch.where(depth_pt != 0., depth_scaled, torch.zeros_like(depth_scaled))
        
            depth_video[i] = depth_scaled.int().unsqueeze(dim=-1).expand(-1, -1, 3)
            albedo_video[i] = torch.from_numpy(albedo.copy())

        
        #np.save( os.path.join(save_path, save_name, f'{save_name}_{angle_deg}_color'), color)
        #np.save( os.path.join(save_path, save_name, f'{save_name}_{angle_deg}_depth'), depth)
        #np.save( os.path.join(save_path, save_name, f'{save_name}_{angle_deg}_albedo'), albedo)
        #np.save( os.path.join(save_path, save_name, f'{save_name}_{angle_deg}_campose'), camera_pose)

        #plt.imshow(color)
        #plt.savefig(f'{save_name}_color_{angle_i}.png', bbox_inches='tight')
        #plt.clf()
        #plt.show()

        #plt.imshow(depth)
        #plt.show()
        
        #plt.imshow(albedo)
        #plt.show()
        
    
    #np.save( os.path.join(save_path, save_name, f'{save_name}_projection'), node_cam.camera.get_projection_matrix())
    #print(node_cam.camera.get_projection_matrix())
    
    if need_video:
        final_video = torch.cat([color_video, depth_video], dim=2)
        torchvision.io.write_video( os.path.join(save_path, save_name, f'{save_name}_rendervideo_color.mp4'), color_video, fps=30)
        torchvision.io.write_video( os.path.join(save_path, save_name, f'{save_name}_rendervideo_depth.mp4'), depth_video, fps=30)
    
    


if __name__ == '__main__':
    # headless rendering
    os.environ['PYOPENGL_PLATFORM']='egl'
    source_folder = '02691156'

    # steps
    steps = 180

    file_list = ['0676', '0775', '1314', '0411', '0447', '1441', '0993', '0671']
    
    for frame_id in file_list:
        file_name = glob.glob(f'*{frame_id}*.ply')[0]
        os.system(f'python ~/.dev_apps/simplemesh/simplemesh.py --input {file_name} -n.85 --output {file_name[:-4]+"_norm.ply"}')
        pcd = trimesh.load(file_name[:-4]+"_norm.ply")
	
        pcd_pyr = pyrender.Mesh.from_points(pcd.vertices, colors=pcd.colors)
    
    
        render_one([pcd_pyr], steps=steps, save_name=file_name[:-4]+'_render', save_path='videos', resolution=(512, 512), need_video=True)
