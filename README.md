1. python 38以上最好
2. open3D 0.11 0.10库才能用
3. troch和cuda 一定要对应，否则loss无法安装
4. loss 文件下setop.py  {cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},}修改在运行
5. - Train
    ```
    python custom_train.py --root your_datapath/CustomData --train_npts 2048 
    # Note: train_npts depends on your dataset
    ```
- Evaluate
    ```
    # Evaluate, infer_npts depends on your dataset
    python custom_evaluate.py --root your_datapath/CustomData --infer_npts 2048 --checkpoint work_dirs/models/checkpoints/test_min_loss.pth --cuda
    
    # Visualize, infer_npts depends on your dataset
    python custom_evaluate.py --root your_datapath/CustomData --infer_npts 2048 --checkpoint work_dirs/models/checkpoints/test_min_loss.pth --show
    ```
  6. custom_evaluate.py 里面执行先用open3d ipc配准一次，只用模型进行调整
