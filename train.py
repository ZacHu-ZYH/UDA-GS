import argparse
import os
import timeit
import datetime

import cv2
from loss_functions import adv_loss, CORAL, kl_js, mmd, mutual_info, cosine, pairwise_dist,mttl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform
import random
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
import glob
from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d


from utils import transformsgpu
from utils.helpers import colorize_mask
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader_4modal, get_data_path_4modal
from data.augmentations import *

import PIL
import matplotlib.pyplot as plt
from torchvision import transforms
import json
import time
import wandb
from Models import U_Net,AttU_Net,NestedUNet,TransUNet

# wandb.init(project='multi-seg')

# config = wandb.config
# config.learning_rate = 0.01

start = timeit.default_timer()
# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA-GS")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='./configs/configUDA.json',
                        help='Path to the config file (default: config.json)')

    parser.add_argument("-r", "--resume", type=str, default='',
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default="Run1",
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

class softmax_dice(nn.Module):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    def __init__(self):
        super(softmax_dice, self).__init__()
        self.ce = CrossEntropy2d()
        
    def forward(self, output, target,weight):
        # target[target == 4] = 3 
        # output = output.cuda()
        target = target.long()
        celoss = self.ce(output, target,weight).cuda()
        loss0 = Dice(output[:, 0, ...], (target == 0).float())
        # import pdb
        # pdb.set_trace()
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 3).float())

        # D0, D1, D2, D3 = dice_score(output, target)
        return loss1 + loss2 + loss3 + loss0+celoss
    
def loss_calc(pred, label,weight):
    criterion = softmax_dice()
    return criterion(pred, label,weight)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):

    ################################Choose the backbone##########################
    ema_model = AttU_Net(12,4)
    #############################################################################
    

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model.cuda()

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model.cuda()

def getaxis(image1, image2, label1, label2,x_set,y_set,scale): # mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters
    #img1 mixed-image img2 target or input image lab1 mixed-lab
    # print(image1.shape,image2.shape,label1.shape,label2.shape)
    img1 = image1.cpu().detach().numpy()
    img2 = image2.cpu().detach().numpy()
    lab1 = label1.cpu().detach().numpy()
    img1 = np.transpose(img1,(1,2,0)).astype(np.float32)
    img2 = np.transpose(img2,(1,2,0)).astype(np.float32)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)/255
    # print(np.max(img1),np.max(img2))
    img_roi_pre = (img2-np.min(img2))/(np.max(img2)-np.min(img2))*255
    # cv2.imshow('mix1_roi',img1)
    # cv2.waitKey(0)
    masked_img = img1.copy()
    masked_img[lab1==0]=0
    mask_roi = cv2.boundingRect(lab1)
    img2_roi = cv2.boundingRect(np.array(img_roi_pre,np.uint8))
    mask = masked_img[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]]
    mask = cv2.resize(mask,(img2_roi[2]//scale,img2_roi[3]//scale))
    mask = (mask-np.min(img2))/(np.max(img2)-np.min(img2))
    ROI = img2_roi[0]+x_set,img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set,img2_roi[1]+y_set+mask.shape[1]
    return ROI

def GliomaMix(image1, image2, label1, label2,x_set,y_set,scale): # mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters

    img1 = image1.cpu().detach().numpy()
    img2 = image2.cpu().detach().numpy()
    lab1 = label1.cpu().detach().numpy()
    img1 = np.transpose(img1,(1,2,0)).astype(np.float32)
    img2 = np.transpose(img2,(1,2,0)).astype(np.float32)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)/255
    img_roi_pre = (img2-np.min(img2))/(np.max(img2)-np.min(img2))*255
    masked_img = img1.copy()
    masked_img[lab1==0]=0
    mask_roi = cv2.boundingRect(lab1)
    img2_roi = cv2.boundingRect(np.array(img_roi_pre,np.uint8))
    mask = masked_img[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]]
    mask = cv2.resize(mask,(img2_roi[2]//scale,img2_roi[3]//scale))
    mask = (mask-np.min(img2))/(np.max(img2)-np.min(img2))
    
    mix1 = img2.copy()
    lab_roi = lab1[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]]
    lab_roi = cv2.resize(lab_roi,(img2_roi[2]//scale,img2_roi[3]//scale),interpolation=cv2.INTER_NEAREST)
    mix1_lab = np.zeros((mix1.shape[0],mix1.shape[1]),np.uint8)

    
    mix1_roi = mix1[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]]
    
    mix1_lab[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]]=lab_roi
    mix1_roi[mask>0.5] = mask[mask>0.5]
    mix1[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]] = mix1_roi
    mix1 = cv2.blur(mix1, (5, 5))
    mix1 = (mix1-np.min(mix1))/(np.max(mix1)-np.min(mix1))*255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mix1 = clahe.apply(np.array(mix1,np.uint8))
    
    mix1 = cv2.cvtColor(mix1, cv2.COLOR_GRAY2RGB)
    mix1 = np.expand_dims(mix1,0)
    
    mix1_lab = np.expand_dims(mix1_lab,0)
    mix1 = np.transpose(mix1,(0,3,1,2))

    out_img = torch.Tensor(mix1)
    out_lbl = torch.Tensor(mix1_lab)
    return out_img, out_lbl

def GliomaMix2(image1, image2, label1, label2,x_set,y_set,scale):
    img1 = image1.cpu().detach().numpy()
    img2 = image2.cpu().detach().numpy()
    lab1 = label1.cpu().detach().numpy()
    lab2 = label2.cpu().detach().numpy()
    img1 = np.transpose(img1,(1,2,0)).astype(np.float32)
    img2 = np.transpose(img2,(1,2,0)).astype(np.float32)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)/255
    img_roi_pre = (img2-np.min(img2))/(np.max(img2)-np.min(img2))*255
    masked_img = img1.copy()
    masked_img[lab1==0]=0
    mask_roi = cv2.boundingRect(lab1)
    img2_roi = cv2.boundingRect(np.array(img_roi_pre,np.uint8))
    mask = masked_img[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]]
    mask = cv2.resize(mask,(img2_roi[2]//scale,img2_roi[3]//scale))
    mask = (mask-np.min(img2))/(np.max(img2)-np.min(img2))
    
    mix1 = img2.copy()
    lab_roi = lab1[mask_roi[1]:mask_roi[1]+mask_roi[3],mask_roi[0]:mask_roi[0]+mask_roi[2]]
    lab_roi = cv2.resize(lab_roi,(img2_roi[2]//scale,img2_roi[3]//scale),interpolation=cv2.INTER_NEAREST)
    mix1_lab = np.zeros((mix1.shape[0],mix1.shape[1]),np.uint8)
    
    mix1_roi = mix1[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]]
    

    mix1_lab[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]]=lab_roi
    mix1_lab = cv2.bitwise_or(mix1_lab, np.array(lab2,np.uint8))
    mix1_roi[mask>0.5] = mask[mask>0.5]
    mix1[img2_roi[0]+x_set:img2_roi[0]+x_set+mask.shape[0],img2_roi[1]+y_set:img2_roi[1]+y_set+mask.shape[1]] = mix1_roi
    mix1 = cv2.blur(mix1, (5, 5))
    mix1 = (mix1-np.min(mix1))/(np.max(mix1)-np.min(mix1))*255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mix1 = clahe.apply(np.array(mix1,np.uint8))
    
    mix1 = cv2.cvtColor(mix1, cv2.COLOR_GRAY2RGB)
    mix1 = np.expand_dims(mix1,0)
    
    mix1_lab = np.expand_dims(mix1_lab,0)
    mix1 = np.transpose(mix1,(0,3,1,2))
    out_img = torch.Tensor(mix1)
    out_lbl = torch.Tensor(mix1_lab)

    return out_img, out_lbl
def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filelist = glob.glob(os.path.join(checkpoint_dir,'*.pth'))
        if filelist:
            os.remove(filelist[0])
        filename = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['model'])
        else:
            ema_model.load_state_dict(checkpoint['model'])

    return iteration, model, optimizer, ema_model

def main():
    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus)
        else:
            unlabeled_loss =  MSELoss2d()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropy2d(), device_ids=gpus)
        else:

            unlabeled_loss = softmax_dice()

    cudnn.enabled = True
    # create network
    ################################Choose the backbone##########################
    model = AttU_Net(12,4)
    #############################################################################

    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    cudnn.benchmark = True
    target_loader = get_loader_4modal('xinhua')
    target_path = get_data_path_4modal('xinhua')

    if random_crop:
        data_aug = Compose([RandomCrop_city(input_size)])
    else:
        data_aug = None

    #data_aug = Compose([RandomHorizontallyFlip()])
    target_dataset = target_loader(target_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)


    np.random.seed(random_seed)
    targetloader = data.DataLoader(target_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    targetloader_iter = iter(targetloader)


    #source domain

    source_loader = get_loader_4modal('BraTs')
    source_path = get_data_path_4modal('BraTs')
    if random_crop:
        data_aug = Compose([RandomCrop_gta(input_size)])
    else:
        data_aug = None

    #data_aug = Compose([RandomHorizontallyFlip()])
    #######################################################&order1##################################################
    source_dataset = source_loader(source_path, list_path = './data/train_mix.txt', augmentations=data_aug, img_size=(384,384), mean=IMG_MEAN)
    sourceloader = data.DataLoader(source_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    sourceloader_iter = iter(sourceloader)
    
    
    ######################################################&order2##################################################
    
    source_mix_dataset = source_loader(source_path, list_path = './data/train_mix.txt', augmentations=data_aug, img_size=(384,384), mean=IMG_MEAN)
    sourcemixloader = data.DataLoader(source_mix_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    sourcemixloader_iter = iter(sourcemixloader)
    ###################################################################################################################

    # optimizer for segmentation network
    learning_rate_object =     (config['training']['learning_rate'])
    

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(),
                        lr=learning_rate)
            # MAX_STEP = int(1e10)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-6)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)


    optimizer.zero_grad()
    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    accumulated_loss_u = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        
        model.train()

        loss_u_value = 0
        loss_l_value = 0
        loss_2_value = 0
        loss_mmd_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(sourceloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourceloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourceloader_iter = iter(sourceloader)
            batch = next(sourceloader_iter)

        if random_flip:
            weak_parameters={"flip":random.randint(0,1)}
        else:
            weak_parameters={"flip": 0}


        images, T1,T1c,T2,labels, _, _ = batch

        images = images.float().cuda()
        T1 = T1.float().cuda()
        T1c = T1c.float().cuda()
        T2 = T2.float().cuda()
        labels = labels.long().cuda()
        
        input_4modal = torch.cat([images,T1,T1c,T2],1)
        pred = model(input_4modal)[0]
        L_l = loss_calc(pred,labels,weight=None) # Cross entropy loss for labeled data
        
        

        try:
            batch = next(sourcemixloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(sourcemixloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            sourcemixloader_iter = iter(sourcemixloader)
            batch = next(sourcemixloader_iter)
            
            
        image_mix, mix_T1,mix_T1c,mix_T2,label_mix, _, _ = batch

        image_mix = image_mix.cuda()
        label_mix = label_mix.cuda()
        mix_T1 = mix_T1.cuda()
        mix_T1c = mix_T1c.cuda()
        mix_T2 = mix_T2.cuda()

        MixMask0_lam = 0.9
        
        if train_unlabeled:
            try:
                batch_target = next(targetloader_iter)
                if batch_target[0].shape[0] != batch_size:
                    batch_target = next(targetloader_iter)
            except:
                targetloader_iter = iter(targetloader)
                batch_target = next(targetloader_iter)
            images_target,target_T1,target_T1c,target_T2, _, _ = batch_target
            images_target = images_target.float()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_target)
            
            images_targetT1 = target_T1.float()
            inputs_u_w_T1, _ = weakTransform(weak_parameters, data = images_targetT1)
            
            images_targetT1c = target_T1c.float()
            inputs_u_w_T1c, _ = weakTransform(weak_parameters, data = images_targetT1c)
            
            images_targetT2 = target_T2.float()
            inputs_u_w_T2, _ = weakTransform(weak_parameters, data = images_targetT2)
            #inputs_u_w = inputs_u_w.clone()
            # logits_u_w = interp(ema_model(inputs_u_w)[0])
            input_4modal_target = torch.cat([inputs_u_w,inputs_u_w_T1,inputs_u_w_T1c,inputs_u_w_T2],1).cuda()
            logits_u_w = ema_model(input_4modal_target)[0]
            logits_u_w, _ = weakTransform(weak_parameters, data = logits_u_w.detach())
            
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
            
            x_set = 50+random.randint(1,50)
            y_set = 50+random.randint(1,50)
            scale = 2+random.randint(1,2)
            inputs_u_s0, targets_u0 = GliomaMix(image_mix[0], images_target[0], label_mix[0],labels[0],x_set,y_set,scale)
            inputs_t_s0, targets_t0 = GliomaMix2(image_mix[0], images[0], label_mix[0], labels[0],x_set,y_set,scale)
            inputs_u_s1, targets_u1 = GliomaMix(image_mix[1], images_target[1], label_mix[1],labels[1],x_set,y_set,scale)
            inputs_t_s1, targets_t1 = GliomaMix2(image_mix[1], images[1], label_mix[1],labels[1],x_set,y_set,scale)
           
            
            inputs_u_s0_T1, targets_u0_T1 = GliomaMix(mix_T1[0], images_targetT1[0], label_mix[0],labels[0],x_set,y_set,scale)
            inputs_t_s0_T1, targets_t0_T1 = GliomaMix2(mix_T1[0], T1[0], label_mix[0], labels[0],x_set,y_set,scale)
            inputs_u_s1_T1, targets_u1_T1 = GliomaMix(mix_T1[1], images_targetT1[1], label_mix[1],labels[1],x_set,y_set,scale)
            inputs_t_s1_T1, targets_t1_T1 = GliomaMix2(mix_T1[1], T1[1], label_mix[1],labels[1],x_set,y_set,scale)
            
            inputs_u_s0_T1c, targets_u0_T1c = GliomaMix(mix_T1c[0], images_targetT1c[0], label_mix[0],labels[0],x_set,y_set,scale)
            inputs_t_s0_T1c, targets_t0_T1c = GliomaMix2(mix_T1c[0], T1c[0], label_mix[0], labels[0],x_set,y_set,scale)
            inputs_u_s1_T1c, targets_u1_T1c = GliomaMix(mix_T1c[1], images_targetT1c[1], label_mix[1],labels[1],x_set,y_set,scale)
            inputs_t_s1_T1c, targets_t1_T1c = GliomaMix2(mix_T1c[1], T1c[1], label_mix[1],labels[1],x_set,y_set,scale)
            
            inputs_u_s0_T2, targets_u0_T2 = GliomaMix(mix_T2[0], images_targetT2[0], label_mix[0],labels[0],x_set,y_set,scale)
            inputs_t_s0_T2, targets_t0_T2 = GliomaMix2(mix_T2[0], T2[0], label_mix[0], labels[0],x_set,y_set,scale)
            inputs_u_s1_T2, targets_u1_T2 = GliomaMix(mix_T2[1], images_targetT2[1], label_mix[1],labels[1],x_set,y_set,scale)
            inputs_t_s1_T2, targets_t1_T2 = GliomaMix2(mix_T2[1], T2[1], label_mix[1],labels[1],x_set,y_set,scale)
            
            mix_target_image = inputs_u_s0.cpu().detach().numpy()
            
            inputs_u_s = torch.cat((inputs_u_s0, inputs_u_s1))
            inputs_t_s = torch.cat((inputs_t_s0, inputs_t_s1))
            
            mix_target_image_T1 = inputs_u_s0_T1.cpu().detach().numpy()
            inputs_u_s_T1 = torch.cat((inputs_u_s0_T1, inputs_u_s1_T1))
            inputs_t_s_T1 = torch.cat((inputs_t_s0_T1, inputs_t_s1_T1))
            
            mix_target_image_T1c = inputs_u_s0_T1c.cpu().detach().numpy()
            inputs_u_s_T1c = torch.cat((inputs_u_s0_T1c, inputs_u_s1_T1c))
            inputs_t_s_T1c = torch.cat((inputs_t_s0_T1c, inputs_t_s1_T1c))
            
            mix_target_image_T2 = inputs_u_s0_T2.cpu().detach().numpy()
            inputs_u_s_T2 = torch.cat((inputs_u_s0_T2, inputs_u_s1_T2))
            inputs_t_s_T2 = torch.cat((inputs_t_s0_T2, inputs_t_s1_T2))
    
            ROI1_target = getaxis(image_mix[0], images_target[0], label_mix[0],labels[0],x_set,y_set,scale)
            ROI2_target = getaxis(image_mix[1], images_target[1], label_mix[1],labels[1],x_set,y_set,scale)
            
            ROI1 = getaxis(image_mix[0], images[0], label_mix[0],labels[0],x_set,y_set,scale)
            ROI2 = getaxis(image_mix[1], images[1], label_mix[1],labels[1],x_set,y_set,scale)
            
            input_4modal_unlabeled = torch.cat([inputs_u_s,inputs_u_s_T1,inputs_u_s_T1c,inputs_u_s_T2],1).cuda()
            
            # x1 = torch.nn.ModuleList(model.children())
            # x2 = len(x1)
            # dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer
            p1,p2 = model(input_4modal_unlabeled)
            # mark = 'source'
            # plot_kernels(dr.features, i_iter, mark,7, cmap="rainbow")
            ap = nn.AdaptiveAvgPool2d((1,1))
            logits_u_s = interp(p1)
            gs = ap(p2).flatten(1)
            p2 = interp(p2)

            f_source = MixMask0_lam * p2
            f_source = ap(f_source).flatten(1)
            
            input_4modal_target = torch.cat([inputs_t_s,inputs_t_s_T1,inputs_t_s_T1c,inputs_t_s_T2],1).cuda()
            # x1 = torch.nn.ModuleList(model.children())
            # x2 = len(x1)
            # dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer
            pt1,pt2 = model(input_4modal_target)
            # mark = 'target'
            # plot_kernels(dr.features, i_iter, mark,7, cmap="rainbow")
            logits_t_s = interp(pt1)
            gt = ap(pt2).flatten(1)
            pt2 = interp(pt2)
            f_target = MixMask0_lam * pt2
            f_target = ap(f_target).flatten(1)
            # loss_feature =  adv_loss.adv(f_source, f_source, input_dim=f_source.shape[1], hidden_dim=32)
            # loss_global =  adv_loss.adv(gs, gt, input_dim=gs.shape[1], hidden_dim=32)
            loss_feature = cosine(f_source,f_target)

            loss_global = cosine(gs,gt)
            
            loss_multi_task = mttl(p2,pt2,x_set,y_set,scale)
            loss_cos = 0.1*loss_global + 0.1*loss_feature + loss_multi_task*0.00005
        
    
    
            targets_u = torch.cat((targets_u0, targets_u1)).cuda().float()
            targets_t = torch.cat((targets_t0, targets_t1)).cuda().float()
    
            lam = 0.9
            L_l2 = loss_calc(logits_t_s, labels.cuda(),weight=None) * (1-lam) + lam * loss_calc(logits_t_s,targets_t,weight=None)

    
            if consistency_loss == 'MSE':   
                    unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                    #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                    L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s,  targets_u,weight=lam2*pixelWiseWeight)
                + consistency_weight * unlabeled_loss(logits_u_s,  targets_u_w,weight=(1-lam2)*pixelWiseWeight)
            loss = L_l + L_u +  L_l2 + loss_cos
        else:
            loss = L_l

        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            
            if train_unlabeled:
                loss_2_value += L_l2.item()
                loss_u_value += L_u.item()
                loss_mmd_value += loss_cos.item()
        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)
        if i_iter % 100 == 0 :
            if train_unlabeled:
                # wandb.log({"loss_l": L_l,"L_u": L_u,"L_l2": L_l2,"loss_cos": loss_cos})
                print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}, loss_2 = {4:.3f}, lambda = {5:.3f}, loss_cos = {6:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value, loss_2_value,lam,loss_mmd_value))
            else:
                # wandb.log({"loss_l": L_l})
                print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}'.format(i_iter, num_iterations, loss_l_value))
        if i_iter % val_per_iter == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)
    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']

    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
    num_classes=4
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label']

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    # train_unlabeled = False
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0,1,2)[:args.gpus]
    # gpus = '2'

    main()

