from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer, GlobalDiscriminator, LocalDiscriminator
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
    '--nepoch', type=int, default=25, help='number of epochs to train for')
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

netG = PointNetDenseCls(device = device, feature_transform=opt.feature_transform)
localD = LocalDiscriminator(k = 2, device = device)
globalD = GlobalDiscriminator(k = 2, device = device)

if opt.model != '':
    netG.load_state_dict(torch.load(opt.model))

optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizerD = optim.Adam(list(globalD.parameters())+list(localD.parameters()), lr=0.001, betas=(0.9, 0.999))

scheduler = optim.lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.5)

netG.to(device)
localD.to(device)
globalD.to(device)

criterion = distChamfer
#Dcriterion = nn.BCELoss()
Dcriterion = F.nll_loss

real_label = 1
fake_label = 0

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in (enumerate(dataloader, 0)):
        points, target, mask = data # Nx4 or Nx3
        points = points.transpose(2, 1) # 4xN
        points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        b_size = points.shape[0]
        
        mask_ = mask.unsqueeze(2).repeat(1,1,3)
        mask__ = ~mask_
        mask__ = mask__.to(device, dtype = torch.float32)
        mask_ = mask_.to(device, dtype = torch.float32)
                
        
        optimizerD.zero_grad()
        
        localD = localD.train()
        globalD = globalD.train()
        
        
        ###### train D ######
        
        label_real =  torch.stack((torch.zeros(b_size),torch.ones(b_size)), dim = 1).to(device)
        label_fake =  torch.stack((torch.ones(b_size),torch.zeros(b_size)), dim = 1).to(device)
        
        target_mask = mask__ * target
        target_mask = target_mask[target_mask.sum(dim = 2) != 0].view(b_size, -1, 3)
        
        target, target_mask = target.transpose(2, 1).contiguous(), target_mask.transpose(2,1).contiguous()

        output_g = globalD(target)[:,1]
        output_l = localD(target_mask)[:,1]
                
        print(output_g.shape)
        print(output_l.shape)
        print(label_real.shape)
        
        errD_real_g = Dcriterion(output_g, label_real)
        errD_real_l = Dcriterion(output_l, label_real)
        
        errD_real = errD_real_g + errD_real_l
        errD_real.backward()
        
        pred = netG(points)
        
        pred = (pred * mask__) + (target * mask_)
                
        pred_mask = pred * mask__
        pred_mask = pred_mask[pred_mask(dim = 2) != 0].view(b_size, -1, 3)
        
        pred, pred_mask = pred.transpose(2, 1).contiguous(), pred_mask.transpose(2,1).contiguous()
        
        output_g = globalD(pred.detach())[:,0]
        output_l = localD(pred_mask.detach())[:,0]
        
        errD_fake_g = Dcriterion(output_g, label_fake)
        errD_fake_l = Dcriterion(output_l, label_fake)
        
        errD_fake = errD_fake_g + errD_fake_l
        errD_fake.backward()
        
        errD = errD_real + errD_fake
                
        optimizerD.step()

        ###### train G ######
        
        optimizerG.zero_grad()
        
        netG = netG.train()
        
        output_g = globalD(pred)[:,1]
        output_l = localD(pred_mask)[:,1]
                
        errG_g = Dcriterion(output_g, label_real)
        errG_l = Dcriterion(output_l, label_real)
        
        errG = errG_g + errG_l

        pred, target = pred.transpose(2, 1).contiguous(), target.transpose(2, 1).contiguous()
        
        dist1, dist2 = criterion(pred, target)
        loss = (torch.mean(dist1)) + (torch.mean(dist2)) + errG
        
        loss.backward()

        
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        optimizerG.step()
        
        print('[%d: %d/%d] train loss: %f ' % (epoch, i, num_batch, loss.item()))
        
        
        

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, mask = data
            points = points.transpose(2, 1)
            points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            b_size = points.shape[0]
            
            localD = localD.eval()
            globalD = globalD.eval()
            
            ###### eval D ######
            
            label_real =  torch.stack((torch.zeros(b_size),torch.ones(b_size)), dim = 1).to(device)
            label_fake =  torch.stack((torch.ones(b_size),torch.zeros(b_size)), dim = 1).to(device)
            
            mask_ = mask.unsqueeze(2).repeat(1,1,3)
            mask__ = ~mask_
            mask__ = mask__.to(device, dtype = torch.float32)
            mask_ = mask_.to(device, dtype = torch.float32)
            
            target_mask = mask__ * target
            target_mask = target_mask[target_mask.sum(dim = 3) != 0]

            output_g = globalD(target).view(-1)
            output_l = localD(target_mask).view(-1)

            errD_real_g = Dcriterion(output_g, label_real)
            errD_real_l = Dcriterion(output_l, label_real)

            errD_real = errD_real_g + errD_real_l

            pred = netG(points)
            pred = (pred * mask__) + (target * mask_)

            pred_mask = pred * mask__
            pred_mask = pred_mask[pred_mask.sum(dim = 3) != 0 ]

            output_g = globalD(pred.detach()).view(-1)
            output_l = localD(pred_mask.detach()).view(-1)

            errD_fake_g = Dcriterion(output_g, label_fake)
            errD_fake_l = Dcriterion(output_l, label_fake)

            errD_fake = errD_fake_g + errD_fake_l

            errD = errD_real + errD_fake


            ###### eval G ######

            netG = netG.eval()

            output_g = globalD(pred).view(-1)
            output_l = localD(pred_mask).view(-1)

            errG_g = Dcriterion(output_g, label_real)
            errG_l = Dcriterion(output_l, label_real)

            errG = errG_g + errG_l

            dist1, dist2 = criterion(pred, target)
            loss = (torch.mean(dist1)) + (torch.mean(dist2)) + errG
            
            print('[%d: %d/%d] %s D_loss: %f, G_loss: %f ' % (epoch, i, num_batch, blue('test'), errD.item(), loss.item()))
    scheduler.step()
    torch.save(netG.state_dict(), '%s/com_model_G_%f_%d.pth' % (opt.outf, loss.item(), epoch))
    torch.save(localD.state_dict(), '%s/com_model_localD_%f_%d.pth' % (opt.outf, errD.item(), epoch))
    torch.save(globalD.state_dict(), '%s/com_model_globalD_%f_%d.pth' % (opt.outf, errD.item(), epoch))