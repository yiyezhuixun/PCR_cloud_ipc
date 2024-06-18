import argparse
import os.path

import numpy as np
import open3d as o3d
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import CustomData
from models import IterativeBenchmark, icp
from metrics import compute_metrics, summary_metrics, print_metrics
from utils import npy2pcd, pcd2npy,npy2pcd_normal


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--root', required=True, help='the data path')
    parser.add_argument('--infer_npts', type=int, required=True,
                        help='the points number of each pc for training')
    parser.add_argument('--in_dim', type=int, default=6,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--checkpoint', default='',
                        help='the path to the trained checkpoint')
    parser.add_argument('--method', default='benchmark',
                        help='choice=[benchmark, icp]')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use the cuda')
    parser.add_argument('--show', action='store_true',
                        help='whether to visualize')
    args = parser.parse_args()
    return args

def evaluate_benchmark(args, test_loader):
    model = IterativeBenchmark(in_dim=args.in_dim,
                               niters=args.niters,
                               gn=args.gn)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))

        # model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
            if args.cuda:
                ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                    ref_cloud.permute(0, 2, 1).contiguous())
            toc = time.time()
            dura.append(toc - tic)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

            if args.show:
                ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
                src_cloud = torch.squeeze(src_cloud).cpu().numpy()
                pred_ref_cloud = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
                pcd1 = npy2pcd(ref_cloud, 0)
                pcd2 = npy2pcd(src_cloud, 1)
                pcd3 = npy2pcd(pred_ref_cloud, 2)
                o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def evaluate_icp(args, test_loader):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, (ref_cloud, src_cloud, gtR, gtt,ref_normals,sef_normals) in tqdm(enumerate(test_loader)):
        if args.cuda:
            ref_cloud, src_cloud, gtR, gtt ,ref_normals,sef_normals= ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda(),ref_normals.cuda(), sef_normals.cuda()

        ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
        src_cloud = torch.squeeze(src_cloud).cpu().numpy()
        ref_normals = torch.squeeze(ref_normals).cpu().numpy()
        sef_normals = torch.squeeze(sef_normals).cpu().numpy()
        # r_1 = npy2pcd(src_cloud)
        # r_2 = npy2pcd(ref_cloud)
        # r_1.paint_uniform_color([1,0,0])
        # r_2.paint_uniform_color([0,1,0])
        # o3d.visualization.draw_geometries([r_1, r_2])
        tic = time.time()
        R, t, pred_ref_cloud = icp(npy2pcd(src_cloud), npy2pcd(ref_cloud))
        print("R2:",R)
        print("T2:",t)
        toc = time.time()
        R = torch.from_numpy(np.expand_dims(R, 0)).to(gtR)
        t = torch.from_numpy(np.expand_dims(t, 0)).to(gtt)
        dura.append(toc - tic)

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
        pcd1 = npy2pcd_normal(ref_cloud, ref_normals,0)
        pcd2 = npy2pcd_normal(src_cloud, sef_normals,1)
        pcd3 = pred_ref_cloud
        pcd3.paint_uniform_color([0, 0, 1])
        if os.path.exists('./pcd1.pcd') or os.path.exists('./pcd1.pcd') or os.path.exists('./pcd1.pcd'):
            os.remove('./pcd1.pcd')
            os.remove('./pcd2.pcd')
            os.remove('./pcd3.pcd')
            print("dele is ok ")
        o3d.io.write_point_cloud('./pcd1.pcd', pcd1, write_ascii=True)  # ascii编码
        o3d.io.write_point_cloud('./pcd3.pcd', pcd3, write_ascii=True)
        o3d.io.write_point_cloud('./pcd2.pcd', pcd2, write_ascii=True)
        o3d.visualization.draw_geometries([pcd3, pcd1])

        # if args.show:
        #     pcd1 = npy2pcd(ref_cloud, 0)
        #     pcd2 = npy2pcd(src_cloud, 1)
        #     pcd3 = pred_ref_cloud
        #     o3d.visualization.draw_geometries([pcd1,pcd2 ,pcd3])
        #     o3d.io.write_point_cloud('./pcd1', pcd1, write_ascii=True)  # ascii编码
        #     o3d.io.write_point_cloud('./pcd3', pcd3, write_ascii=True)
        #     o3d.io.write_point_cloud('./pcd2', pcd2, write_ascii=True)

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


if __name__ == '__main__':
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = config_params()

    test_set = CustomData(args.root, args.infer_npts, False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    if args.method == 'benchmark':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_benchmark(args, test_loader)
        print_metrics(args.method,
                      dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    elif args.method == 'icp':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_icp(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    else:

        raise NotImplementedError