import cv2
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data, model_zoo

from data import get_data_path_4modal, get_loader_4modal

from PIL import Image
from Models import U_Net,AttU_Net,NestedUNet,TransUNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def colorize_mask(mask):
    # mask: numpy array of the mask
    # new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask = Image.fromarray(mask.astype(np.uint8))
    # new_mask.putpalette(palette)

    return new_mask

def max_area(mask_sel):
    
    contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     
    #找到最大区域并填充
     
    area = []
     
    for j in range(len(contours)):
     
        area.append(cv2.contourArea(contours[j]))
     
    max_idx = np.argmax(area)
     
    max_area = cv2.contourArea(contours[max_idx])
     
    for k in range(len(contours)):
     
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    
    return mask_sel


def main():
    """Create the model and start the evaluation process."""

    gpu0 = '0,1,2'
    model = AttU_Net(12,4)

    checkpoint = torch.load('./saved/')

    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=gpu0)
        model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()


    target_loader = get_loader_4modal('FeTS15')
    target_path = get_data_path_4modal('FeTS15')
    target_dataset = target_loader(target_path, is_transform=True, augmentations=None, img_size=input_size, img_mean = IMG_MEAN)
    targetloader = data.DataLoader(target_dataset,
                        batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    for index, batch in enumerate(targetloader):
        # try:
        image, T1,T1c,T2,_, name = batch
        # size = size[0]

        with torch.no_grad():
            input_4modal = torch.cat([image,T1,T1c,T2],1)
            output,_  = model(Variable(input_4modal.float()).cuda())

            output = output.cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)*63
            # output = max_area(output)
            output_col = colorize_mask(output)
            output_col.save(r'%s/%s' % (save_dir, str(name[0])))

        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

if __name__ == '__main__':
    # args = get_arguments()

    # config = torch.load(args.model_path)['config']

    num_classes = 4
    input_size = (384,384)

    save_dir = ''
    main()
