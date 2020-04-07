# import ssl
import math
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data import get_train_test_loader


if __name__ == '__main__':
    # ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader = get_train_test_loader('dataset', batch_size=16, num_workers=4)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 100)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = 0.0001
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.5)
    epochs = 64
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, targets = data
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    torch.save(model.state_dict(),
               os.path.join('static', 'models', 'model_' + str(int(math.log2(weight_decay / 0.0001) + 1)) + '_gpu.pth'))
    model.to('cpu')
    torch.save(model.state_dict(),
               os.path.join('static', 'models', 'model_' + str(int(math.log2(weight_decay / 0.0001) + 1)) + '_cpu.pth'))

