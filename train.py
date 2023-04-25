import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import time
import cv2
import random
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from torchvision import transforms
from dtn import DTN
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.autograd import Variable

from dataset import MyDataset

# ['17,37,103', '75,86,173', '180,42,42', '0,0,0', '137,0,0'] # r,g,b
exp_colors = [['001', [103, 37, 17]], 
             ['002', [173, 86, 75]], 
             ['003', [42, 42, 180]], 
             ['004', [0, 0, 0]], 
             ['005', [0, 0, 137]]] # [index, [b, g, r]]


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default= './pretrained-checkpoint/resnet50-0676ba61.pth', help='pretrained weights path')
    parser.add_argument('--data', type=str, default='./carla-datasets/datasets/train', help='train dataset path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', type=int, default=448, help='train, val image size (pixels)')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--fixencoder', type=bool, default=False, help='if fix encoder weights')

    parser.add_argument('--save_dir', type=str, default='./run/train/weight', help='where to save the trained model weight')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def seed_torch(seed=42):
    # fix python && numpy  random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # ban hash random
    np.random.seed(seed)
    # fix torch CPU && GPU random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # cudnn
    """
    torch.backends.cudnn.benchmark=True :
    cuDNN使用非确定性算法寻找最高效算法。
    将会让程序在开始时花费一点额外时间,为整个网络的每个卷积层搜索最适合它的卷积实现算法,进而实现网络的加速、增加运行效率。

    torch.backends.cudnn.benchmark = False ：禁用基准功能会导致 cuDNN 确定性地选择算法，可能以降低性能为代价。 
    保证gpu每次都选择相同的算法,但是不保证该算法是deterministic的。
    """
    torch.backends.cudnn.benchmark = False
    """
    每次返回的卷积算法将是确定的，即默认算法。
    配合上设置 Torch 的随机种子为固定值,可以保证每次运行网络的时候相同输入的输出是固定的。     
    """
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


'''
ref : array([0.01715409, 0.01746695, 0.01839768]  [0.07684436, 0.07797393, 0.08145176]
exp : array([0.01072918, 0.00556702, 0.01379947]  [0.04316137, 0.02239505, 0.05551254]
ren : array([0.01558956, 0.01473877, 0.01702777]  [0.07186571, 0.06890791, 0.07681201]
'''
def getState(train_dataset):
    '''
    Compute mean and variance for training data
    :param train_data: Dataset class
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros((3, 3))
    std = torch.zeros((3, 3))
    for (ref, exp, ren) in train_loader:
        for d in range(3):
            mean[0][d] += ref[:, d, :, :].mean()
            std[0][d] += ref[:, d, :, :].std()
            mean[1][d] += exp[:, d, :, :].mean()
            std[1][d] += exp[:, d, :, :].std()
            mean[2][d] += ren[:, d, :, :].mean()
            std[2][d] += ren[:, d, :, :].std()
    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))
    return list(mean.numpy()), list(std.numpy())


def getBinaryTensor(imgTensor, boundary = 0):
    zero = torch.zeros_like(imgTensor)
    one = torch.ones_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)


def plot(loss_list, error_list):
    plt.subplot(3, 1, 1)
    loss_x = range(0, len(loss_list))
    plt.plot(loss_x, loss_list, '.-')
    plt_title = 'train loss - epoch'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(3, 1, 3)
    error_x = range(0, len(error_list))
    plt.plot(error_x, error_list, '.-')
    plt_title = 'train mse error - epoch'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('MSE error')

    plt.draw()  
    plt.savefig('./run/train/loss_MSEerror.png')
    plt.show()


def main(opt):
    # define autuencoder model
    device = torch.device(opt.device)
    model = DTN(ae_type='ResNet50').to(device)
    summary(model, [[opt.batchsize, 3, opt.imgsz, opt.imgsz], [opt.batchsize, 3, opt.imgsz, opt.imgsz]])

    # load data pair  list [[ref,exp,ren],[ref,exp,ren]] 
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((opt.imgsz, opt.imgsz)),
                                          # transforms.RandomHorizontalFlip(),                       
                                          ])

    print('******start loading*******')
    """
    train_dataloder: 
    for ref, exp, ren in train_dataloder:
        ref/exp/ren torch.Size(batch_size, 3, 224, 224)
    """
    train_dataset = MyDataset(data_dir=opt.data, exp_colors=exp_colors, transform=train_transform)
    print('Train Dataset have [%d] samples' % len(train_dataset))
    train_dataloder = DataLoader(dataset=train_dataset,
                                 batch_size=opt.batchsize,
                                 shuffle=True,
                                 num_workers=0)
    print('******load data success******')
    
    # load pretrained resnet50 model 267
    pretrained_dict = torch.load(opt.checkpoint, map_location=device)
    print("load pretrained model success")

    pretrained_dict_file = open('pretrained_dict.txt', 'w').close()
    for k in pretrained_dict.keys():
        with open(r'./pretrained_dict.txt', 'a') as f:
            f.write(k)
            f.write('\r\n')

    model_dict_file = open('model_dict.txt', 'w').close()
    model_dict = model.state_dict()
    for k in model_dict.keys():
        with open(r'./model_dict.txt', 'a') as f:
            f.write(k)
            f.write('\r\n')

    # 1. filter out unnecessary keys 
    # e.i filter fc.weight fc.bias
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if opt.fixencoder:
        fix_length = len(pretrained_dict.keys())
        all_length = len(model.state_dict().keys())
        for index, k in enumerate(model_dict.keys()):
            if index < fix_length:
                model.state_dict()[k].requires_grad = False
        
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    
    # Loss and Optimizer
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')  # gradient descent based on the binary cross-entropy loss
    criterion = nn.MSELoss(reduction='mean')
    MSEerror = nn.MSELoss(reduction='mean')  

    optimizer = torch.optim.Adam(params, lr=1e-4)   # betas=(0.9, 0.999)

    model.train()
    print('start training!!!')
    loss_list = []
    error_list = []

    for epoch in range(opt.epochs):
        total_loss = 0
        total_error = 0
        t0 = time.time()
        for batch_index, (ref, exp, ren) in enumerate(train_dataloder):
            # ref, exp, ren  torch.Size(batch_size, 3, 224, 224)
            ref = ref.to(device)
            exp = exp.to(device)
            ren = ren.to(device)

            # reset gradient
            optimizer.zero_grad()

            # forward
            feature, rendered= model(ref, exp)

            # calculate loss
            loss = criterion(rendered, ren)
            total_loss += loss.item()
            # calculate mse error
            error = MSEerror(rendered, ren)
            total_error += error.item()
            
            # backward calculate gradient
            loss.backward()
            
            # optimize update parameters of net
            optimizer.step()

            # show every 10 epoch
            if (batch_index + 1) % 10 == 0:
                print('     Epoch [%d/%d], Batch [%d] Loss: %.4f, MSE: %.4f' % (epoch + 1, opt.epochs, batch_index + 1, loss.item(), error.item()))
        
        loss_list.append(total_loss/(batch_index + 1))
        error_list.append(total_error/(batch_index + 1))

        print('Epoch [%d/%d], consume time: [%.3f s], Loss:%.4f, Error:%.4f' % 
              (epoch + 1, opt.epochs, (time.time() - t0), total_loss/(batch_index + 1), total_error/(batch_index + 1)))
    
    os.makedirs(opt.save_dir, exist_ok=True)
    save_weight_path = os.path.join(opt.save_dir, 'dtn_resnet50_NoRandomHorizontalFlip_{}.pth'.format(opt.epochs))
    torch.save(model.state_dict(), save_weight_path)
    # display the result
    plot(loss_list, error_list)


if __name__ == "__main__":
    seed_torch(seed=42)  
    opt = parse_opt()
    main(opt)

    

    
