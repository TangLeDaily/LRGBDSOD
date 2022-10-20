from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
from data import *
from torch import optim
from util import *

# model name
from rootmodel.ResNet34_DR34 import *
model = LRGBDSOD()
test_dataset_name = ['DUT', 'NJUD', 'NLPR']
# ['DUT', 'NJUD', 'NLPR', 'SSD', 'STEREO', 'LFSD', 'RGBD135']

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
# wandb and project
parser.add_argument("--use_wandb", default=True, action="store_true")
parser.add_argument("--Project_name", default="LRGBDSOD_V1", type=str) # wandb Project name
parser.add_argument("--This_name", default="ResNet34_DR34_FFM", type=str) # wandb run name & model save name path
parser.add_argument("--wandb_username", default="tangle", type=str)
# dataset 文件夹要以/结尾
parser.add_argument("--train_image_root", default='datasets/train_ori/train_images/', type=str, help="train root path")
parser.add_argument("--train_gt_root", default='datasets/train_ori/train_masks/', type=str, help="train root path")
parser.add_argument("--train_depth_root", default='datasets/train_ori/train_depth/', type=str, help="train root path")
parser.add_argument("--trainsize", default=256, type=int)
parser.add_argument("--test_root_path", default='datasets/test_data/', type=str, help="test root path")
# train setting
parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--cuda_id", default=4, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=10000, type=int)
parser.add_argument("--batchSize", default=16, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--threads", default=8, type=int)
# other setting
parser.add_argument("--dataset_pin_memory", default=True, action="store_true")
parser.add_argument("--dataset_drop_last", default=True, action="store_true")
parser.add_argument("--dataset_shuffle", default=True, action="store_true")
parser.add_argument("--test_save_epoch", default=10, type=int)
parser.add_argument("--decay_loss_epoch", default=50, type=int)
parser.add_argument("--decay_loss_ratio", default=0.8, type=float)
opt = parser.parse_args()



def main():
    global model, opt
    if opt.use_wandb:
        wandb.init(project=opt.Project_name, name=opt.This_name, entity=opt.wandb_username)
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    if cuda:
        torch.cuda.set_device(opt.cuda_id)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    # 利用显存换取浮点训练加速
    # cudnn.benchmark = True

    print("===> Loading datasets")
    train_loader = get_loader(opt.train_image_root, opt.train_gt_root, opt.train_depth_root, batchsize=opt.batchSize, trainsize=opt.trainsize)
    # test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)

    print("===> Setting loss")
    criterion = torch.nn.BCEWithLogitsLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Do Resume Or Skip")
    # model = get_yu(model, "checkpoints/over/TSALSTM_ATD/model_epoch_212_psnr_27.3702.pth")

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # 训练
        train(optimizer, model, criterion, epoch, train_loader)
        # 测试和保存
        if (epoch+1) % opt.test_save_epoch == 0:
            save_mae, save_Em, save_Sm, save_Fm = test(model, epoch, optimizer.param_groups[0]["lr"])
            save_checkpoint(model, epoch, optimizer.param_groups[0]["lr"], save_mae, save_Em, save_Sm, save_Fm)
        # 降低学习率
        if (epoch+1) % opt.decay_loss_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] *= opt.decay_loss_ratio

def train(optimizer, model, criterion, epoch, train_loader):
    global opt
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    avg_loss = AverageMeter()
    if opt.cuda:
        model = model.cuda()
    for iteration, batch in enumerate(train_loader):
        images, gts, depths = batch
        if opt.cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            depths = torch.cat([depths, depths, depths], dim=1)
        out = model(images, depths)
        loss = criterion(out, gts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            if opt.use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg})
            print('epoch_iter_{}_loss is {:.10f}'.format(iteration, avg_loss.avg))

def test(model, epoch, lr, savename="DUT"):
    print(" -- Start eval --")
    global opt
    model.eval()
    if not os.path.exists("checkpoints/{}/".format(opt.This_name)):
        os.makedirs("checkpoints/{}/".format(opt.This_name))
    log_write("checkpoints/{}/Test_log.txt".format(opt.This_name), "===> Epoch_{}:".format(epoch))
    save_mae, save_Em, save_Sm, save_Fm = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for dataset_name in test_dataset_name:
            test_dataloader = get_test_dataloader(opt.test_root_path, dataset_name, 256)
            T_mae = AverageMeter()
            T_Fm = AverageMeter()
            T_Em = AverageMeter()
            T_Sm = AverageMeter()
            for i in range(test_dataloader.size):
                image, gt, depth, name = test_dataloader.load_data()
                if opt.cuda:
                    image = image.cuda()
                    depth = depth.cuda()
                    depth = torch.cat([depth, depth, depth], dim=1)
                    gt = gt.cuda()
                out = model(image, depth)
                out = out.sigmoid()
                T_mae.update(MAE(out, gt))
                T_Em.update(Em(out, gt))
                T_Sm.update(Sm(out, gt))
                T_Fm.update(Fm(out, gt))
            avg_mae = T_mae.avg
            avg_Em = T_Em.avg
            avg_Sm = T_Sm.avg
            avg_Fm = T_Fm.avg
            if dataset_name == savename:
                save_mae = avg_mae
                save_Em = avg_Em
                save_Sm = avg_Sm
                save_Fm = avg_Fm
            if opt.use_wandb:
                wandb.log({'{}_MAE'.format(dataset_name): avg_mae,
                           '{}_Em'.format(dataset_name): avg_Em,
                           '{}_Sm'.format(dataset_name): avg_Sm,
                           '{}_Fm'.format(dataset_name): avg_Fm,
                           'Epoch':epoch})
            print("===> lr:{:.8f} dataset_name:{} Em:{:.4f} Sm:{:.4f} Fm:{:.4f} MAE:{:.4f}".format(lr, dataset_name, avg_Em, avg_Sm, avg_Fm, avg_mae))
            log_write("checkpoints/{}/Test_log.txt".format(opt.This_name), "lr:{:.8f} dataset_name:{} Em:{:.4f} Sm:{:.4f} Fm:{:.4f} MAE:{:.4f}".format(lr, dataset_name, avg_Em, avg_Sm, avg_Fm, avg_mae))
    return save_mae, save_Em, save_Sm, save_Fm


def save_checkpoint(model, epoch, lr, save_mae, save_Em, save_Sm, save_Fm):
    global opt
    model_folder = "checkpoints/{}/".format(opt.This_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "EP_{}_LR_{:.8f}_Em_{:.4f}_Sm_{:.4f}_Fm_{:.4f}_MAE_{:.4f}.pth".format(epoch, lr, save_Em, save_Sm, save_Fm, save_mae)
    torch.save({'model': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()