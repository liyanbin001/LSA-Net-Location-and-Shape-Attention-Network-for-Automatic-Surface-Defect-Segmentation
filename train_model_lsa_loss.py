import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import model_lsa_loss

from utils.calaverage import AverageMeter, str2bool
from data_loader import Dataset3
from utils.metrics import iou_score
from utils import losses
ARCH_NAMES = model_lsa_loss.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='LocSegnets',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet_back_denoising)')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dataset4_split',
                        help='dataset name')
    parser.add_argument('--dataset_train', default='train',
                        help='dataset name')
    parser.add_argument('--dataset_test', default='test',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.bmp',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.bmp',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'loss5': AverageMeter(),
                  'iou': AverageMeter()}

    pbar = tqdm(total=len(train_loader))
    model.train()
    loss_mse = torch.nn.MSELoss()
    loss_l1 = torch.nn.L1Loss()
    loss_kld = torch.nn.KLDivLoss()

    for input, target,label, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        label = label.cuda()
        output, Lout1, Pout1, out_add1, targetM, labelM, Lout, Pout = model(input, target, label)
        loss1 = criterion(output, target)
        loss2 = criterion(Lout1, targetM)
        loss3 = criterion(Pout1, targetM)
        loss4 = criterion(out_add1, labelM)
        # loss5 = 1-100*loss_kld(Lout.softmax(dim=1).log(), Pout.softmax(dim=1))
        # print(loss5)
        loss5 = 1 / loss_mse(Lout, Pout)
        # loss = 1.4 * loss1 + 0.2*(loss2+loss3 + loss5)

        loss = 1.4 * loss1 + 0.4 * loss4 + 0.2 * loss5
        #loss = 1.4 * loss1 + 0.6 * loss4
        iou = iou_score(output, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['loss5'].update(loss5.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('loss5', avg_meters['loss5'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('loss5', avg_meters['loss5'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'loss5': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    loss_mse = torch.nn.MSELoss()
    loss_l1 = torch.nn.L1Loss()
    loss_kld = torch.nn.KLDivLoss()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
        best_iou = 0
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))

        for input, target,label, meta in val_loader:
            input = input.cuda()
            target = target.cuda()
            label = label.cuda()
            output, Lout1, Pout1, out_add1, targetM, labelM, Lout, Pout = model(input, target, label)
            loss1 = criterion(output, target)
            loss2 = criterion(Lout1, targetM)
            loss3 = criterion(Pout1, targetM)
            loss4 = criterion(out_add1, labelM)
            # loss5 = loss_kld(Lout.softmax(dim=1).log(), Pout.softmax(dim=1))
            # loss5 = criterion(out_add1, labelM)
            # loss5 = 1-100*loss_kld(Lout.softmax(dim=1).log(), Pout.softmax(dim=1))
            # print(loss5)
            loss5 = 1 / loss_mse(Lout, Pout)
            # loss = 1.4 * loss1 + 0.2 * (loss2 + loss3 + loss5)
            # loss5 = 1/loss_mse(Lout,Pout)
            loss = 1.4 * loss1 + 0.4 * loss4 + 0.2 * loss5
            #loss = 1.4 * loss1 + 0.6 * loss4
            iou = iou_score(output, target)
            pre = output
            pre = torch.sigmoid(pre).cpu().numpy()

            out1 = Lout1
            out1 = torch.sigmoid(out1).cpu().numpy()

            out_add1 = out_add1
            out_add1 = torch.sigmoid(out_add1).cpu().numpy()

            # for i in range(len(pre)):
            #     for c in range(config['num_classes']):
            #         path_write0 = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg')
            #         pre_image = (pre[i, c] * 255).astype('uint8')
            #         cv.imwrite(path_write0,pre_image)
            #         path_write0 = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png')
            #         pre_image = (out1[i, c] * 255).astype('uint8')
            #         cv.imwrite(path_write0,pre_image)
            #         path_write0 = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.tif')
            #         pre_image = (out_add1[i, c] * 255).astype('uint8')
            #         cv.imwrite(path_write0,pre_image)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['loss5'].update(loss5.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('loss5', avg_meters['loss5'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('loss5', avg_meters['loss5'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if config['name'] is None:
        config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = model_lsa_loss.__dict__[config['arch']]()

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], config['dataset_train'],'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    img_ids = glob(os.path.join('inputs', config['dataset'],config['dataset_test'],'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        # transforms.RandomRotation((90), expand=True),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        # transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        # transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset3(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'],config['dataset_train'],'images'),
        mask_dir=os.path.join('inputs', config['dataset'], config['dataset_train'],'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset3(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'],config['dataset_test'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'],config['dataset_test'],'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('loss5', []),
        ('iou', []),
        ('val_loss', []),
        ('val_loss5', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['loss5'].append(train_log['loss5'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_loss5'].append(val_log['loss5'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
