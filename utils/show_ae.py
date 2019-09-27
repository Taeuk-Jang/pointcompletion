from __future__ import print_function
#from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import sys
sys.path.append("..")
from pointnet.dataset import ShapeNetDataset
from pointnet.model import Autoencoder
import matplotlib.pyplot as plt
from pointnet.plyfile import PlyData
from pyntcloud import PyntCloud
import pandas as pd

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='completion/com_model_ae_0.002717_0.pth', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='/home/cdi0/data/shape_net_core_uniform_samples_2048_split/', help='dataset path')
parser.add_argument('--device', type=str, default='cuda:0', help='dataset path')

#parser.add_argument('--class_choice', type=str, default='', help='class choice')


opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    dir=opt.dataset,
    train='test',
    )
device = opt.device
idx = opt.idx
print(d.lst[idx])
print("model %d/%d" % (idx, len(d)))
point, target, mask = d[idx]
print(point.shape, target.shape)
#point_np = point.numpy()



state_dict = torch.load(opt.model, map_location='cpu')
classifier = Autoencoder(device = device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()

input_cloud = PyntCloud(pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=point[:,:3],
    columns=["x", "y", "z"]))
input_cloud.to_file("input.ply")

target_cloud = PyntCloud(pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=target,
    columns=["x", "y", "z"]))
target_cloud.to_file("target.ply")


point = torch.from_numpy(point.transpose(1, 0)).to(device, dtype=torch.float)
target = torch.from_numpy(target).to(device, dtype=torch.float)
mask = torch.from_numpy(mask)
mask = mask.view(1, 2048, 1)

#point = Variable(point.transpose(1, 0).to(device, dtype=torch.float))
#target = Variable(target.transpose(1, 0).to(device, dtype=torch.float))

point, target = point.unsqueeze(0), target.unsqueeze(0)

#target = Variable(target.view(1, target.size()[0], target.size()[1]))

pred, conf = classifier(point)


#points = points.transpose(2, 1)
#points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

#classifier = classifier.eval()
#pred, _, _ = classifier(points)
#pred = classifier(points)

mask_ = mask.repeat(1,1,3)
mask__ = ~mask_
mask__ = mask__.to(device, dtype = torch.float32)
mask_ = mask_.to(device, dtype = torch.float32)

print(pred.shape)
print(target.shape)
print(mask_.shape)
print(mask__.shape)
print('mean confidence value : %f'% conf.min().item())

pred = (pred * mask__) + (target * mask_)

pred = pred.squeeze(0).cpu().detach().numpy()
                 
cloud = PyntCloud(pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=pred.reshape(2048,-1),
    columns=["x", "y", "z"]))
cloud.to_file("output.ply")

#print(pred_choice.size())
#pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
#showpoints(point_np, gt, pred_color)
