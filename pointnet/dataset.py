import torch.utils.data as data
import os
import os.path
#from plyfile import PlyData, PlyElement
from plyfile import PlyData
import numpy as np

def load_ply(file_name, with_faces=False, with_color=False):
    
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    
    return points

def load_list(root, train = 'train'):
    input_dir = []
    rootdir = root
    #rootdir = '/home/cdi0/data/shape_net_core_uniform_samples_2048_split/'

    if train =='train':
        rootdir = os.path.join(rootdir, train)

        for dirs in os.listdir(rootdir):
            if dirs == 'train_0':
                target_dir = os.path.join(rootdir, dirs)
            elif dirs.startswith('train'):
                input_dir.append(os.path.join(rootdir, dirs))

    else:
        rootdir = os.path.join(rootdir, 'test') 

        for dirs in os.listdir(rootdir):
            if dirs == 'test_0':
                target_dir = os.path.join(rootdir, dirs)
            elif dirs.startswith('test'):
                input_dir.append(os.path.join(rootdir, dirs))

    input_dir.sort()

    input_data_list = []
    target_data_list = []

    
    for i in input_dir:
        lst = []
        for dirpath, dirnames, filenames in os.walk(i):
            for filename in [f for f in filenames if f.endswith(".ply")]:
                lst.append(os.path.join(dirpath, filename))
        lst.sort()
        input_data_list.append(lst)

    for dirpath, dirnames, filenames in os.walk(target_dir):
            for filename in [f for f in filenames if f.endswith(".ply")]:
                target_data_list.append(os.path.join(dirpath, filename))

    target_data_list.sort()

    input_set_list = []
    for i in range(len(input_data_list)):
        lst = []
        for j in range(len(input_data_list[i])):
            lst.append((input_data_list[i][j], target_data_list[j]))
        input_set_list.append(lst)
        
    return input_set_list

class ShapeNetDataset(data.Dataset):
    def __init__(self, dir, train = 'train', n_points = 2048, augmentation = False, stage = 0, opt = None):
        
        self.root = dir
        self.loader = load_ply
        self.opt = opt
        self.train = train
        
        lst = []
        l = load_list(dir, self.train)
        for i in range(stage+1):
            lst = lst + l[-1]
            
        self.lst = lst
        self.loader = load_ply
        
    def __getitem__(self, idx):
    
        input_pcd, target_pcd = self.lst[idx]
        input_pcd = self.loader(input_pcd)
        target_pcd = self.loader(target_pcd)
        input_pnt = input_pcd.shape[0]
        mask = np.isin(target_pcd,input_pcd)
        m = np.all(mask, axis = 1)
        
        t = np.zeros((target_pcd.shape[0],4))
        t[:,3] = m
        
        n = 0
        for i in range(len(m)):
            if m[i] == 1 and n<input_pnt:
                t[i,:3] = input_pcd[n]
                n +=1
            else:
                t[i,:3] = np.random.randn(1,3) / 5
                
        input_pcd = t
        
        return input_pcd, target_pcd, m
    
    
    def __len__(self):
        return len(self.lst)
    