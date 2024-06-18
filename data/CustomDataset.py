# import numpy as np
# import os
# import torch
# from torch.utils.data import Dataset
#
# from utils import readpcd
# from utils import pc_normalize, random_select_points, shift_point_cloud, \
#     jitter_point_cloud, generate_random_rotation_matrix, \
#     generate_random_tranlation_vector, transform
#
# class CustomData(Dataset):
#     def __init__(self, root, npts, train=True):
#         super(CustomData, self).__init__()
#         dirname = 'train_data' if train else 'val_data'
#         path = os.path.join(root, dirname)
#         self.train = train
#         self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
#         self.npts = npts
#
#     def __getitem__(self, item):
#         # file = self.files[2]
#         # if item+1 < len(self.files) - 1:
#         #     file1 = self.files[item]
#         #     file2 =  self.files[item+1]
#         # else:
#         #     file1 = self.files[item]
#         #     file2 = self.files[0]
#
#         # file = self.files[item]
#         # print(file1,file2)
#         # print("item",item)
#         file1 ="CustomData/val_data/0527_09_pc - Cloud - Cloud - Cloud.pcd"
#         ref_cloud = readpcd(file1, rtype='npy')
#         ref_cloud = random_select_points(ref_cloud, m=self.npts)
#         ref_cloud = pc_normalize(ref_cloud)
#         # file1 = self.files[3]
#         file2 = "CustomData/val_data/0530_06_pc - Cloud - Cloud.pcd"
#         # print(file1)
#         src_cloud = readpcd(file2, rtype='npy')
#         src_cloud = random_select_points(src_cloud, m=self.npts)
#         src_cloud = pc_normalize(src_cloud)
#         R, t = generate_random_rotation_matrix(-10, 10), \
#                generate_random_tranlation_vector(-0.1, -0.1)
#         # src_cloud = transform(ref_cloud, R, t)
#         if self.train:
#             ref_cloud = jitter_point_cloud(ref_cloud)
#             src_cloud = jitter_point_cloud(src_cloud)
#         return ref_cloud, src_cloud, R, t
#
#     def __len__(self):
#         return len(self.files)


import numpy as np
import os
import torch
from torch.utils.data import Dataset
import open3d as o3d
from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform
from models.icp import fast_icp
class CustomData(Dataset):
    def __init__(self, root, npts, train=True):
        super(CustomData, self).__init__()
        dirname = 'train_data' if train else 'val_data'
        path = os.path.join(root, dirname)
        self.train = train
        self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        self.npts = npts

    def __getitem__(self, item):
        if item+1 < len(self.files) - 1:
            file1 = self.files[item]
            file2 =  self.files[item+1]
        else:
            file1 = self.files[0]
            file2 = self.files[1]

        # file = self.files[item]
        # print(file1,file2)
        # print("item",item)
        # file1 = 'CustomData/val_data/0617_23_pc_1_0.0.pcd'
        # file2 = "CustomData/val_data/0617_31_pc_1_0.0.pcd"
        voxel_size = 1
        source_temp, target_temp = fast_icp(file1, file2,voxel_size)

        # ref_cloud,ref_normals = readpcd(file1, rtype='npy')
        ref_cloud,ref_normals =  np.asarray(source_temp.points).astype(np.float32),np.asarray(source_temp.normals).astype(np.float32)
        print("len1:",len(ref_cloud))
        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        ref_cloud = pc_normalize(ref_cloud)
        # file1 = self.files[3]
        # print(file1)
        # src_cloud,sef_normals = readpcd(file2, rtype='npy')
        src_cloud, src_normals = np.asarray(target_temp.points).astype(np.float32),np.asarray(target_temp.normals).astype(np.float32)
        print("len2:", len(src_cloud))
        src_cloud = random_select_points(src_cloud, m=self.npts)
        src_cloud = pc_normalize(src_cloud)

        # r_1 = o3d.io.read_point_cloud(file1)
        # r_2 = o3d.io.read_point_cloud(file2)
        # r_1.paint_uniform_color([1, 0, 0])
        # r_2.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([r_1, r_2])




        R, t = generate_random_rotation_matrix(-0.01, 0.01), \
               generate_random_tranlation_vector(-0.05, 0.05)
        # src_cloud = transform(ref_cloud, R, t)
        if self.train:
            ref_cloud = jitter_point_cloud(ref_cloud)
            src_cloud = jitter_point_cloud(src_cloud)
        return ref_cloud, src_cloud, R, t,ref_normals, src_normals

    def __len__(self):
        return len(self.files)


# import numpy as np
# import os
# import torch
# from torch.utils.data import Dataset
#
# from utils import readpcd
# from utils import pc_normalize, random_select_points, shift_point_cloud, \
#     jitter_point_cloud, generate_random_rotation_matrix, \
#     generate_random_tranlation_vector, transform
#
# class CustomData(Dataset):
#     def __init__(self, root, npts, train=True):
#         super(CustomData, self).__init__()
#         dirname = 'train_data' if train else 'val_data'
#         path = os.path.join(root, dirname)
#         self.train = train
#         self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
#         self.npts = npts
#
#     def __getitem__(self, item):
#         # file = self.files[2]
#         if item+1 < len(self.files) - 1:
#             file1 = self.files[item]
#             file2 =  self.files[item+1]
#         else:
#             file1 = self.files[item]
#             file2 = self.files[0]
#
#         # file = self.files[item]
#         # print(file1,file2)
#         # print("item",item)
#         # file = 'CustomData/val_data/pcd1.pcd'
#         ref_cloud = readpcd(file1, rtype='npy')
#         ref_cloud = random_select_points(ref_cloud, m=self.npts)
#         ref_cloud = pc_normalize(ref_cloud)
#         # file1 = self.files[3]
#         # file1 = "CustomData/val_data/pcd3.pcd"
#         # print(file1)
#         src_cloud = readpcd(file2, rtype='npy')
#         src_cloud = random_select_points(src_cloud, m=self.npts)
#         src_cloud = pc_normalize(src_cloud)
#         R, t = generate_random_rotation_matrix(-5, 5), \
#                generate_random_tranlation_vector(-0.1, 0.1)
#         # src_cloud = transform(ref_cloud, R, t)
#         if self.train:
#             ref_cloud = jitter_point_cloud(ref_cloud)
#             src_cloud = jitter_point_cloud(src_cloud)
#         return ref_cloud, src_cloud, R, t
#
#     def __len__(self):
#         return len(self.files)
