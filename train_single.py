from transformers import SegformerForSemanticSegmentation
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp

import dataloader
import os
from datetime import datetime

import numpy as np
import sys
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from models import UNet11
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import argparse

from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics import JaccardIndex

import importlib.util

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", help="Directory path of config files")
parser.add_argument("config_name", help="Name of config to import, e.g. default_config if the config file is default_config.py")
args = parser.parse_args()

# import config
config_namespace = f"configs.{args.config_name}"
config_fp = os.path.join(args.config_dir, f"{args.config_name}.py")
spec = importlib.util.spec_from_file_location(config_namespace, config_fp)
config = importlib.util.module_from_spec(spec)
sys.modules[config_namespace] = config
spec.loader.exec_module(config)
cfg = config.Config()


output_folder = f"{cfg.output_folder}_{datetime.now().strftime('%m-%d-%Y_%H%M')}"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

log_file = open(os.path.join(output_folder, "log.txt"), "w")
def log(s):
	print(s)
	log_file.write(s+"\n")
	log_file.flush()

s = f"Using config file {args.config_name} \n" \
	+ f"config path: {args.config_dir}\n {cfg}"
log(s)

if cfg.unet and cfg.segformer:
    print("You cannot specify both segformer and unet")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_sets = []
val_sets = []

print("Loading data")

def update_metrics(pred, lbl):
    for i in range(pred.size(0)):
        for (name, metric) in pytorch_metrics.items():
            metric.update(pred[i].flatten(), lbl[i].flatten())
    jaccard(pred, lbl)

def compute_pytorch_metrics():
    metric_vals = {}
    for name, metric in pytorch_metrics.items():
        metric_vals[name] = metric.compute()
        metric.reset()

    metric_vals["jaccard"] = jaccard.compute()
    jaccard.reset()

    return metric_vals

def tb_log(epoch, writer, names, metrics):
    for name, metric in zip(names, metrics):
        writer.add_scalar(name, metric, epoch)
    writer.flush()

weights = np.zeros(cfg.num_classes, dtype=np.float32)

for x in os.walk(cfg.data_dir):
    val = False
    test = False

    if debug:
        print("debug")
        if not "03" in x[0] and not "04" in x[0] and not "05" in x[0]:
            continue

    if cfg.organ in x[0]:
        c_lbl = 1
    else:
        continue

    if not os.path.isfile(x[0] + "/image00.png"):
        continue

    for id in cfg.test_ids: #Skip test data
        if id in x[0]:
            test = True
            break
    if test:
        continue #Skip test data

    for id in cfg.val_ids:
        if id in x[0]:
            val = True
            break

    # Create dataloaders
    if val:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, cfg.num_classes, cfg.val_transform,
                                               organ_id=cfg.organ_id, organ_name=cfg.organ, 
                                               p_neg_img=cfg.p_neg_img,
                                               image_size=cfg.image_size)
        val_sets.append(dataset)
    else:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, cfg.num_classes, cfg.train_transform, 
                                               image_size=cfg.image_size, 
                                               aug_method=cfg.aug, k_aug=cfg.k, seed=cfg.seed,sim_score=cfg.sim_score)

        train_sets.append(dataset)
        #Collect frequencies for class weights
        bg_w, p = dataset.get_frequency()
        c_w = p - bg_w 
        weights[0] += bg_w
        weights[c_lbl] += c_w

print('TraingSingle/weights',weights)
n_samples = np.sum(weights)

weights = n_samples/(cfg.num_classes*weights)
weights[weights == np.inf] = 0.1
print(weights)

train_sets = torch.utils.data.ConcatDataset(train_sets)
val_sets = torch.utils.data.ConcatDataset(val_sets)

train_loader = torch.utils.data.DataLoader(train_sets, batch_size=cfg.mini_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=cfg.mini_batch_size, shuffle=False)


writer = SummaryWriter()
writer.add_text("ConfigName", args.config_name, global_step=0)

if cfg.unet:
    model = UNet11(num_classes=cfg.num_classes, pretrained=True)
elif cfg.segformer:
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",num_labels=cfg.num_classes,ignore_mismatched_sizes=True)
else:
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, cfg.num_classes)

model.train()
model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=1e-1)#, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=True)

acc = BinaryAccuracy(device=device)
f1 = BinaryF1Score(device=device)
pc = BinaryPrecision(device=device)
rc = BinaryRecall(device=device)
pytorch_metrics = {"acc": acc, "f1": f1, "pc": pc, "rc": rc}
jaccard = JaccardIndex(task="binary", num_classes=1).to(device)

best_f1 = 0

print("Training model")
for e in range(cfg.epochs):
    optimizer.zero_grad()
    train_batches = 0
    val_batches = 0
    train_loss_sum = 0
    val_loss_sum = 0

    model.train()
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for img, lbl, _ in train_loader:
        sys.exit()
        if img.size(0) < 2:
            continue
        img = img.to(device)
        lbl = lbl.to(device).long()

        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):

            outputs = model(img)
            if cfg.unet:
                out = outputs
            elif cfg.segformer:
                out = nn.functional.interpolate(outputs["logits"], size=img.shape[-2:], mode="bilinear", align_corners=False)
            else:
                out = outputs['out']
            
            loss = criterion(out, lbl)
            
            pred = torch.argmax(out, 1)

            train_loss.append(loss.item())
            train_loss_sum += loss.item()
            train_accuracy.append(torch.sum(pred == lbl).item()/(cfg.mini_batch_size*cfg.image_size[0]*cfg.image_size[1]))

        if cfg.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        train_batches += 1

        if train_batches % cfg.num_mini_batches == 0:
            if cfg.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
    if train_batches % cfg.num_mini_batches != 0:
        if cfg.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    model.eval()
    preds = []

    with torch.no_grad():
        for img, lbl, _ in val_loader:
            img = img.to(device)
            lbl = lbl.to(device).long()

            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                outputs = model(img)

                if cfg.unet:
                    out = outputs
                elif cfg.segformer:
                    out = nn.functional.interpolate(outputs["logits"], size=img.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    out = outputs['out']
                loss = criterion(out, lbl)
                pred = torch.argmax(out, 1)
                preds.append(pred.cpu().numpy())

                update_metrics(pred, lbl)

                val_loss.append(loss.item())
                val_loss_sum += loss.item()
                val_accuracy.append(torch.sum(pred == lbl).item()/(cfg.mini_batch_size*cfg.image_size[0]*cfg.image_size[1]))
                val_batches += 1

    pytorch_metric_vals = compute_pytorch_metrics()
    train_loss_val = train_loss_sum / train_batches if train_batches > 0 else 1
    val_loss_val = val_loss_sum / val_batches if val_batches > 0 else 1
    
    names = ["train_loss", "val_loss", *pytorch_metric_vals.keys()]
    metrics = [train_loss_val, val_loss_val, *pytorch_metric_vals.values()]
    tb_log(e, writer, names, metrics)

    scheduler.step()

    if (e + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(output_folder, "model%04d.th" % e))
        log(f"Epoch {e}: val loss: {metrics[1]} jac: {pytorch_metric_vals['jaccard']}")


    if best_f1 < pytorch_metric_vals['f1']:
        best_epoch = e
        best_f1 = pytorch_metric_vals['f1']
        torch.save(model.state_dict(), os.path.join(output_folder, "model_best.th"))


log_file.write(f"Best f1: {best_f1} (epoch {best_epoch})")
log_file.flush()

writer.flush()
writer.close()
