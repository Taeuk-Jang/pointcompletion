from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append("..")
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
#from emd import EMDLoss

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='completion', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', default = '/home/cdi0/data/shape_net_core_uniform_samples_2048_split/', type=str,  help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', default = False, action='store_true', help="use feature transform")
parser.add_argument('--device', type=str, default='cuda:1', help='gpu device')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    dir=opt.dataset,
    )
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    dir=opt.dataset,
    train='test',
    )
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))


try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'
device = opt.device

classifier = PointNetDenseCls(device = device, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)
#dist =  EMDLoss()
criterion = distChamfer


num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in (enumerate(dataloader, 0)):
        points, target, mask = data
        points = points.transpose(2, 1)
        points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                
        optimizer.zero_grad()
        classifier = classifier.train()
        
        pred = classifier(points)
        
        mask_ = mask.unsqueeze(2).repeat(1,1,3)
        mask__ = ~mask_
        mask__ = mask__.to(device, dtype = torch.float32)
        mask_ = mask_.to(device, dtype = torch.float32)

        pred = (pred * mask__) + (target * mask_)

        
        #loss = F.nll_loss(pred, target)
        #print(pred.shape)
        #print(pred[mask__].shape)
        #pred_ = pred[mask__].view(opt.batchSize,-1,3)
        #target_ = target[mask__].view(opt.batchSize,-1,3)
        
        #cost = dist(target_, target_)
        #loss = torch.sum(cost)
        dist1, dist2 = criterion(pred, target)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        print(loss.shape)
        loss.backward()
        #loss.backward()
        
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        optimizer.step()
        
        print('[%d: %d/%d] train loss: %f ' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, mask = data
            points = points.transpose(2, 1)
            points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            classifier = classifier.eval()
            #pred, _, _ = classifier(points)
            pred = classifier(points)
            
            mask_ = mask.unsqueeze(2).repeat(1,1,3)
            mask__ = ~mask_
            mask__ = mask__.to(device, dtype = torch.float32)
            mask_ = mask_.to(device, dtype = torch.float32)

            pred = (pred * mask__) + (target * mask_)

            mask_ = mask.unsqueeze(2).repeat(1,1,3)
            mask__ = ~mask_

            #loss = F.nll_loss(pred, target)
            #pred_ = pred[mask__].view(opt.batchSize,-1,3)
            #target_ = target[mask__].view(opt.batchSize,-1,3)
            
            dist1, dist2 = criterion(pred, target)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            #cost = dist(pred_, target_)
            #loss = torch.sum(cost)      
            
            print('[%d: %d/%d] %s loss: %f ' % (epoch, i, num_batch, blue('test'), loss.item()))
    scheduler.step()
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, mask = data
    points = points.transpose(2, 1)
    points, target = points.to(device = opt.device), target.to(device = opt.device)
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))