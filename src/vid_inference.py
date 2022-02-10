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
import av

from .MareHabitatDataset import MareHabitatDatasetInference
from .create_frames_lst import create_list

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


def save_frames_to_path(vid, fnums, save_path):   
    counter = 0
    container = av.open(vid)

    for fnum, frame in enumerate(container.decode(video=0)):
        if fnum not in fnums:
            continue

        v_id = vid.split('.')[0].split('/')[-1]
        sv_name = os.path.join(save_path, '{}_{:07d}'.format(v_id, fnum))
        if os.path.isfile(sv_name+'.npy'):
            continue
        
        if counter % 10 == 0:
            print('Saving frame {}'.format(counter))
        img = frame.to_image()
        arr = np.array(img)
        np.save(sv_name, arr)
        counter += 1

    container.close()

    return


def initialize_transforms(input_size):
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


def predict(vid, k, model_weight, batch_size, num_workers, outfile):

    device = torch.device("cpu")
    print('\n\n\n\n\n\nTORCH IS AVAILABLE:::\n\n\n', torch.cuda.is_available())
    # first create list of frames and save it and theframes to tmp directory
    if os.path.exists(TMP_PATH):
        shutil.rmtree(TMP_PATH)
    os.makedirs(FRAMES_HOME)

    create_list(vid, k, FRAMES_LIST_HOME)
    with open(FRAMES_LIST_HOME, 'r') as f:
        full_ids = f.readlines()

    fnums = set()
    
    for id in full_ids:
        print('\n\n\n\nHELLLLLLLOOOOOOO\n\n\n', full_ids)
        fnums.add(int(id.split('_')[-1]))
    fnums = sorted(fnums)

    save_frames_to_path(vid, fnums, FRAMES_HOME)

    # now create the data set, data loader, and model
    model, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))
    data_transforms = initialize_transforms(input_size)
    image_dataset = MareHabitatDatasetInference(FRAMES_LIST_HOME, FRAMES_HOME, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ids = {}
    for samples in dataloader:
        inputs = samples['image'].to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)

        for idx, pred in enumerate(preds):
            ids[samples['id'][idx]] = []
            new_preds = np.array(pred.detach().cpu())
            for id in range(new_preds.shape[0]):
                if new_preds[id] > THRESH:
                    ids[samples['id'][idx]].append(IDX_TO_SUBSTRATE[id])

    df = pd.DataFrame.from_dict(ids, orient='index')
    df = df.sort_index()
    df.to_csv(outfile, header=False)

    shutil.rmtree(TMP_PATH)

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
