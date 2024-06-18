import os

path = r"CustomData/train_data/"
oldlist = os.listdir(path)  # 获得所有文件名列表
newlist = sorted(oldlist, key=lambda x: os.path.getmtime(os.path.join(path, x)))  # 按时间排序的文件名列表
a = 1
for i in newlist:
    os.rename(path + "/" + i, path + "/" +str('train') +str(a) + ".pcd")  # 重命名
    a += 1
