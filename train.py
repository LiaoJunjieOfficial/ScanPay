# import ssl
import math
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data import get_train_test_loader
from engine import train_one_epoch, evaluate


if __name__ == '__main__':
    # ssl._create_default_https_context = ssl._create_unverified_context
    weight_decay = 0.0001
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader = get_train_test_loader('dataset')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 100)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.5)
    epochs = 128
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step(epoch)
        evaluate(model, test_loader, device=device)
    torch.save(model.state_dict(),
               os.path.join('static', 'models', 'model_' + str(int(math.log2(weight_decay / 0.0001) + 1)) + '_gpu.pth'))
    model.to('cpu')
    torch.save(model.state_dict(),
               os.path.join('static', 'models', 'model_' + str(int(math.log2(weight_decay / 0.0001) + 1)) + '_cpu.pth'))
