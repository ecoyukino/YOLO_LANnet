import time
import os
import sys

from tqdm import tqdm

import torch
from dataloader.data_loaders import LaneDataSet
from dataloader.transformers import Rescale
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from utils.cli_helper import parse_args
from utils.average_meter import AverageMeter


import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#usercode--------------------------------
import torch.nn as nn
import lanenet_loss as loss
from lib.models.YOLOP import MCnet
# might want this in the transformer part as well
VGG_MEAN = [103.939, 116.779, 123.68]
#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#usercode-------------------------------------------------
def compute_loss(net_output, binary_label, instance_label):
    k_binary = 0.7
    k_instance = 0.3
    k_dist = 1.0

    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label)
    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = loss.DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001,usegpu=True)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    
    binary_loss = binary_loss * k_binary
    instance_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    total_loss = binary_loss + instance_loss + dist_loss
    out = net_output["binary_seg_pred"]
    iou = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        iou += TP / union
    iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, out, iou
#------------------------------------------------------------------------
def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    val_gt = (image_data[i].cpu().numpy().transpose(1, 2, 0) + VGG_MEAN).astype(np.uint8)
    val_pred = out[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    val_out[:, :, 1] = val_label
    val_gt[val_out == 255] = 255
    # epsilon = 1e-5
    # pix_embedding = pix_embedding[i].data.cpu().numpy()
    # pix_vec = pix_embedding / (np.sum(pix_embedding, axis=0, keepdims=True) + epsilon) * 255
    # pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1, 2, 0)
    # ins_label = instance_label[i].data.cpu().numpy().transpose(0, 1)
    # ins_label = np.repeat(np.expand_dims(ins_label, -1), 3, -1)
    # val_img = np.concatenate((val_gt, pix_vec, ins_label), axis=0)
    # val_img = np.concatenate((val_gt, pix_vec), axis=0)
    # return val_img
    return val_gt

def train(train_loader, model, optimizer, epoch, device):
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    step = 0
    DEVICE = device
    t = tqdm(enumerate(iter(train_loader)), leave=False, total=len(train_loader))
    for batch_idx, batch in t:
        #print("batch = ",batch)
        step += 1
        image_data = Variable(batch[0]).type(torch.FloatTensor).to(DEVICE)
        binary_label = Variable(batch[1]).type(torch.LongTensor).to(DEVICE)
        instance_label = Variable(batch[2]).type(torch.FloatTensor).to(DEVICE)
        image_data = image_data.to(DEVICE)
        # forward pass
        print("----------net_output = model(image_data)----------")
        print("DEVICE = ", type(DEVICE))
        #image_data = image_data.cpu()
        net_output = model(image_data)
        
        # compute loss
        total_loss, binary_loss, instance_loss, out, train_iou = compute_loss(net_output, binary_label, instance_label)

        # update loss in AverageMeter instance
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(train_iou, image_data.size()[0])

        # reset gradients
        optimizer.zero_grad()

        #backpropagate
        total_loss.backward()

        # update weights
        optimizer.step()

        # update batch time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 500 == 0:
            print(
                "Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.2f}|Total loss:{tot:.5f}|Binary loss:{bin:.5f}|Instance loss:{ins:.5f}|IoU:{iou:.5f}".format(
                    ep=epoch + 1,
                    st=step,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    et=batch_time.val,
                    tot=total_losses.avg,
                    bin=binary_losses.avg,
                    ins=instance_losses.avg,
                    iou=train_iou,
                ))
            sys.stdout.flush()
            train_img_list = []
            for i in range(3):
                train_img_list.append(
                    compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, i))
            train_img = np.concatenate(train_img_list, axis=1)
            cv2.imwrite(os.path.join("./output", "train_" + str(epoch + 1) + "_step_" + str(step) + ".png"), train_img)
    return mean_iou.avg


def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch}_checkpoint.pth')
    torch.save(model, save_name)
    print("model is saved: {}".format(save_name))


def main(model,device):
    args = parse_args()
    DEVICE = device
    save_path = args.save

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')
    train_dataset = LaneDataSet(train_dataset_file, transform=transforms.Compose([Rescale((640, 640))]))
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")
    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        train_iou = train(train_loader, model, optimizer, epoch, device)
        if (epoch + 1) % 5 == 0:
            save_model(save_path, epoch, model)
        print(f"Train IoU : {train_iou}")
    return model


def lanenet_train(model,device):
    _model = main(model,device)
    return _model