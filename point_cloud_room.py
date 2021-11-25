# -*- coding: utf-8 -*-
"""
Created on Tue Nov 09 16:28:12 2021

@author: diego

I used the code provided in https://learngeodata.eu/2021/05/14/learn-3d-point-cloud-segmentation-with-python/ for most
of my .ply related code
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import open3d as o3d
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

file_data_path_xyz_corridor=".\objects\jordan4.xyz"#my corridor
file_data_path_kitchen=".\objects\TLS_kitchen.ply"
file_data_path_corridor=r".\objects\my_corridor.ply"
# In[]: load and plot an .xyz file
def load_my_xyz_file(file_path):
    #get data
    point_cloud= np.loadtxt(file_path,delimiter=',')
    #extract colours and coors.
    rgb=point_cloud[:,3:]
    xyz=point_cloud[:,:3]
    #normalize
    xyz = NormalizeData(xyz)
    
    #create fig and plot points
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyz[:,2], xyz[:,0], xyz[:,1], c = rgb/255, s=5)
    ax.set_title('3D object (.xyz)')#, s=1)
    plt.show()

# In[]: load a .ply file 
def load_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    #helps with shading and colour
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=16), fast_normal_computation=True)
    return pcd

# In[]: show the point cloud object
def plot_pcd(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)#standard vals. 
    #get inliners for plotting
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# In[]: get all the segmented objects in a file
def DBSCAN(pcd,min_points=10):
    
    #create groundtruth of groups with a min num of members
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=min_points))
    max_label = labels.max()
    
    #asign colours to each group
    colors = plt.get_cmap("tab20")(labels / (max_label 
    if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #plot objects
    o3d.visualization.draw_geometries([pcd])
    
    
    
# In[]: separate object in the desired amount of segments
def segment_ply_object(pcd,max_plane_num=6):
    segment_params={}#stores parameters for each segment
    segments={}#stores actual segmets

    rest=pcd
    for i in range(max_plane_num):
        colors = plt.get_cmap("tab20")(i)
        segment_params[i], inliers = rest.segment_plane(
        distance_threshold=0.01,ransac_n=3,num_iterations=1000)
        #if len(inliers)<1000:

        segments[i]=rest.select_by_index(inliers)
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)
        print("pass",i+1,"/",max_plane_num,"done. Num of inliners: ",len(inliers))
        
    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_num)]+[rest])

# In[]: create point cloud from rgb image 
#Original code from https://github.com/isl-org/Open3D/issues/2814
def pcd_from_rgb(color_raw_path,depth_raw_path):
    color_raw = o3d.io.read_image(color_raw_path)
    depth_raw = o3d.io.read_image(depth_raw_path)
    color = o3d.geometry.Image(np.array(np.asarray(color_raw)[:, :, :3]).astype('uint8'))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth_raw, convert_rgb_to_intensity=True)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.cpu.pybind.camera.PinholeCameraIntrinsic)


pcd= load_ply_file(file_data_path_corridor)
#segment_ply_object(pcd)
# load_xyz_file(file_data_path)
#pcd_from_rgb()