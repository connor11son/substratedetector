import time
import os
import sys
import psutil
import pickle
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

# constants
NUM_CLASSES = 5
MODEL_NAME = 'resnet18'
FEATURE_EXTRACT = 0
THRESH = 0.5
TMP_PATH = './tmp'
FRAMES_HOME = os.path.join(TMP_PATH, 'frames')
FRAMES_LIST_HOME = os.path.join(TMP_PATH, 'frames_list.txt')
IDX_TO_SUBSTRATE = {0:'Boulder', 1:'Cobble', 2:'Mud', 3:'Rock', 4:'Sand'}
        

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def initialize_transforms(input_size):
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


def prepare_single_batch(model, transforms, arr):
    '''function to return output of single image'''

    if len(arr.shape != 4):
        raise ValueError('Input array should be of dimensions (n, h, w, c)')
    h,w,c = arr[0].shape
    # crop out top and sides
    arr_7575 = arr[:,int(0.25*h):, int(0.125*w):int(0.875*w),:]
    tens = transforms(arr_7575)



def predict(vid, k, model_weight, batch_size, num_workers, outfile):

    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    # now create the model and transforms
    model, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device(device)))
    
    transforms = initialize_transforms()
    for 

    '''df = pd.DataFrame.from_dict(ids, orient='index')
    df = df.sort_index()
    df.to_csv(outfile, header=False)'''


    return

    
if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int)
    p.add_argument('--vid', type=str)
    p.add_argument('--k', type=int, help='run inference on every "kth" frame')
    p.add_argument('--num_workers', type=int)
    p.add_argument('--model_weight', type=str)
    p.add_argument('--outfile', type=str)
    args = p.parse_args()

    batch_size = args.batch_size
    vid = args.vid
    k = args.k
    num_workers = args.num_workers
    model_weight = args.model_weight

    predict(vid, k, model_weight, batch_size, num_workers, outfile)
