import copy
import numpy as np
import open3d as o3d
import time




def icp(source, target):

    max_correspondence_distance = 0.5 # 0.5 in RPM-Net
    init = np.eye(4, dtype=np.float32)
    estimation_method = o3d.registration.TransformationEstimationPointToPoint()

    evaluation = o3d.registration.evaluate_registration(source, target, max_correspondence_distance, init)
    print("evaluation",evaluation)  # 这里输出的是初始位置的 fitness和RMSE

    reg_p2p = o3d.registration.registration_icp(
        source=source,
        target=target,
        init=init,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=estimation_method,
        criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=150)

    )

    transformation = reg_p2p.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    # estimate.paint_uniform_color([1,0,0])
    # target.paint_uniform_color([0,1,0])
    # source.paint_uniform_color([0,0,1])
    # o3d.visualization.draw_geometries([source])
    return R, t, estimate


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    return  source_temp,target_temp

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size,source,target):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud(r"C:\Al\3D\PCReg.PyTorch-main\CustomData/val_data/0617_13_pc_1_0.0.pcd")
    # target = o3d.io.read_point_cloud(r"C:\Al\3D\PCReg.PyTorch-main\CustomData/val_data/0617_24_pc_1_0.0.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                               target_fpfh, voxel_size):
   distance_threshold = voxel_size * 1.5
   print(":: RANSAC registration on downsampled point clouds.")
   print("   Since the downsampling voxel size is %.3f," % voxel_size)
   print("   we use a liberal distance threshold %.3f." % distance_threshold)
   result = o3d.registration.registration_ransac_based_on_feature_matching(
       source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
       o3d.registration.TransformationEstimationPointToPoint(False), 4, [
           o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.registration.CorrespondenceCheckerBasedOnDistance(
               distance_threshold)
       ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
   return result

def fast_icp(source_pa, target_pa,voxel_size):
    source = o3d.io.read_point_cloud(source_pa)
    target = o3d.io.read_point_cloud(target_pa)
    # voxel_size = 0.5
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)
    start = time.time()
    result_fast = execute_global_registration(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print("FAST_ICP",result_fast.transformation)
    R1, t2 = result_fast.transformation[:3, :3], result_fast.transformation[:3, 3]
    print("R1: ",R1)
    print(" t2: ",t2)
    source_temp,target_temp = draw_registration_result(source_down, target_down,
                             result_fast.transformation)
    print("Fast registration result took %.3f sec.\n" % (time.time() - start))
    return source_temp,target_temp





# if __name__ == '__main__':
#     source = o3d.io.read_point_cloud(r"C:\Al\3D\PCReg.PyTorch-main\CustomData/val_data/0617_19_pc_1_0.0.pcd")
#     target = o3d.io.read_point_cloud(r"C:\Al\3D\PCReg.PyTorch-main\CustomData/val_data/0617_06_pc_1_0.0.pcd")
#     voxel_size = 1 # means 5cm for this dataset
#
#
#     source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,source,target)
#
#     start = time.time()
#     result_fast = execute_global_registration(source_down, target_down,
#                                                    source_fpfh, target_fpfh,
#                                                    voxel_size)
#     print("Fast global registration took %.3f sec.\n" % (time.time() - start))
#     print(result_fast)
#     draw_registration_result(source_down, target_down,
#                              result_fast.transformation)

    # result_ransac = execute_global_registration(source_down, target_down,
    #                                            source_fpfh, target_fpfh,
    #                                            voxel_size)
    # print(result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)
