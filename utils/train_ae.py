from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import sys
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import os
sys.path.append("..")
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer, GlobalDiscriminator, LocalDiscriminator, Autoencoder
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter


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
    return P.min(2)[0].mean(1)

def distChamfer_withconf(a, b, c, w):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    d = torch.empty(c.shape).to(device)
    for i in range(bs):
        d[i,:] = c[i, P.min(1)[1][i,:]]
    
    #P.min(1)[0].mean(1) + P.min(2)[0].mean(1)
    
    return (P.min(1)[0] * d - w * torch.log(d) + P.min(2)[0] * c - w * torch.log(c)).mean(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchsize', type=int, default=18, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='completion', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', default = '/home/cdi0/data/shape_net_core_uniform_samples_2048_split/', type=str,  help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', default = False, action='store_true', help="use feature transform")
parser.add_argument('--device', type=str, default='cuda:1', help='gpu device')
parser.add_argument('--multigpu', type=bool, default=False, help='gpu device')
parser.add_argument('--w', type=float, default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
w = opt.w
stage = 0



dataset = ShapeNetDataset(
    dir=opt.dataset, stage = stage
    )
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    dir=opt.dataset, stage = stage,
    train='test',
    )
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

print(len(dataset), len(test_dataset))


try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'
if opt.multigpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = opt.device

ae = Autoencoder(device = device, feature_transform=opt.feature_transform)
#localD = LocalDiscriminator(k = 2, device = device)
#globalD = GlobalDiscriminator(k = 2, device = device)

if opt.model != '':
    ae.load_state_dict(torch.load(opt.model))

optimizerAE = optim.Adam(ae.parameters(), lr=0.0005, betas=(0.9, 0.999))
#optimizerD = optim.Adam(list(globalD.parameters())+list(localD.parameters()), lr=0.0005, betas=(0.9, 0.999))

schedulerAE = optim.lr_scheduler.StepLR(optimizerAE, step_size=20, gamma=0.5)
#schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=20, gamma=0.5)

if opt.multigpu:
    ae = nn.DataParallel(ae)
ae.to(device)
#localD.to(device)
#globalD.to(device)
if opt.multigpu:
    criterion = (distChamfer_withconf)
else:
    criterion = distChamfer_withconf
#Dcriterion = nn.BCELoss()
#Dcriterion = F.nll_loss

real_label = 1.
fake_label = 0.


num_batch = len(dataset) / opt.batchsize
n = int(num_batch/100)
writer = SummaryWriter()
best_loss = np.inf
loss_avg = np.inf

for epoch in range(opt.nepoch):
    if loss_avg < 0.0015:
        stage += 1
        print('switch data to stage %d' % stage)
        
        dataset = ShapeNetDataset(
            dir=opt.dataset, stage = stage
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=int(opt.workers))

        test_dataset = ShapeNetDataset(
            dir=opt.dataset, stage = stage,
            train='test',
            )
        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True)

        print(len(dataset), len(test_dataset))
        
    loss_avg = 0
    for i, data in (enumerate(dataloader, 0)):
        points, target, mask = data
        points = points.transpose(2, 1)
        b_size, _, _ = points.shape
        points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                
        optimizerAE.zero_grad()
        ae = ae.train()
        
        pred, conf = ae(points)
        m = ~mask
        #print(m)
        conf_mask = conf * m.to(device, dtype = torch.float32)
        
        mean_conf = conf_mask[conf_mask != 0].view(b_size, -1).mean(-1).mean(-1)
        conf_mask[conf_mask ==0] = 1
        conf_min = torch.mean(conf_mask.min(-1)[0])
        
        # mask = 0 for to be generated
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

        loss = torch.mean(criterion(pred, target, conf, w))
        loss.backward()
        #loss.backward()
        
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        optimizerAE.step()
        loss_avg += loss.item()/num_batch
        print('[%d: %d/%d] train loss: %f, mean confidence score : %f, min conf soce : %f   ' % (epoch, i, num_batch, loss.item(), mean_conf.item(), conf_min.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, mask = data
            points = points.transpose(2, 1)
            b_size, _, _ = points.shape
            points, target = points.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            ae = ae.eval()
            #pred, _, _ = classifier(points)
            pred, conf = ae(points)
            m = ~mask
            
            conf_mask = conf * m.to(device, dtype = torch.float32)
            
            mean_conf_eval = conf_mask[conf_mask != 0].view(b_size, -1).mean(-1).mean(-1)
            conf_mask[conf_mask ==0] = 1
            conf_min_eval = torch.mean(conf_mask.min(-1)[0])
            
            mask_ = mask.unsqueeze(2).repeat(1,1,3)
            mask__ = ~mask_
            mask__ = mask__.to(device, dtype = torch.float32)
            mask_ = mask_.to(device, dtype = torch.float32)

            pred = (pred * mask__) + (target * mask_)

            #loss = F.nll_loss(pred, target)
            #pred_ = pred[mask__].view(opt.batchSize,-1,3)
            #target_ = target[mask__].view(opt.batchSize,-1,3)
            
             
            loss_eval = torch.mean(criterion(pred, target, conf_mask, w))
            #cost = dist(pred_, target_)
            #loss = torch.sum(cost)      
            
            print('[%d: %d/%d] %s loss: %f, mean confidence score : %f, min conf soce : %f ' % (epoch, i, num_batch, blue('test'), loss.item(), mean_conf_eval.item(), conf_min_eval.item() ))

        if i % 100 ==0:   
            c = int(i / 100)
            
            #writer.add_scalar('errD_real', errD_real.item(), 27 * epoch + n)
            #writer.add_scalar('errD_fake', errD_fake.item(), 27 * epoch + n)
            #writer.add_scalar('errD_loss', errD.item(), 27 * epoch + n)

            #writer.add_scalar('validation errD_real', errD_real_eval.item(), 27 * epoch + n)
            #writer.add_scalar('validation errD_fake', errD_fake_eval.item(), 27 * epoch + n)
            #writer.add_scalar('validation errD_loss', errD_eval.item(), 27 * epoch + n)

            #writer.add_scalar('errG_global', errG_g.item(), 27 * epoch + n)
            #writer.add_scalar('errG_local', errG_l.item(), 27 * epoch + n)
            writer.add_scalar('chamfer_loss', loss.item(), n * epoch + c)
            writer.add_scalar('mean confidence score', mean_conf.item(), n * epoch + c)
            writer.add_scalar('min conf soce', conf_min.item(), n * epoch + c)
            #writer.add_scalar('errG_loss', loss.item(), 27 * epoch + n)

            #writer.add_scalar('validation errG_global', errG_g_eval.item(), 27 * epoch + n)
            #writer.add_scalar('validation errG_local', errG_l_eval.item(), 27 * epoch + n)
            writer.add_scalar('validation chamfer_loss', loss_eval.item(), n * epoch + c)
            writer.add_scalar('mean confidence score', mean_conf_eval.item(), n * epoch + c)
            writer.add_scalar('min conf soce', conf_min_eval.item(), n * epoch + c)
            #writer.add_scalar('validation errG_loss', loss_eval.item(), 27 * epoch + n)

            #for name, param in globalD.named_parameters():
            #    writer.add_histogram(name, param.clone().cpu().data.numpy(), 27 * epoch + n)
            #for name, param in localD.named_parameters():
            #    writer.add_histogram(name, param.clone().cpu().data.numpy(), 27 * epoch + n)
            #for name, param in netG.named_parameters():
            #    writer.add_histogram(name, param.clone().cpu().data.numpy(), 27 * epoch + n)

    schedulerAE.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        if opt.multigpu:
            torch.save(ae.state_dict(), '%s/mult_com_model_ae_%f_%d.pth' % (opt.outf, loss.item(), epoch))
        else:
            torch.save(ae.state_dict(), '%s/com_model_ae_%f_%d.pth' % (opt.outf, loss.item(), epoch))
        #torch.save(localD.state_dict(), '%s/com_model_localD_%f_%d.pth' % (opt.outf, errD.item(), epoch))
        #torch.save(globalD.state_dict(), '%s/com_model_globalD_%f_%d.pth' % (opt.outf, errD.item(), epoch))
    
    