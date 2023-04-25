import os
import cv2
import torch 
import torch.nn as nn
from dtn import DTN
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset

data_dir = './carla-datasets/datasets/test'
batch_size = 4
test_exp_rendered_color = [['001', [225, 225, 225]],
                           ['002', [0, 255, 255]],
                           ['003', [0, 100, 0]],
                           ['004', [19, 69, 139]]]


if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = DTN(ae_type='ResNet50')
    model_dict = torch.load('./run/train/weight/dtn_resnet50_NoRandomHorizontalFlip_100.pth')
    model.load_state_dict(model_dict)

    # load data pair  list [[ref,exp,ren],[ref,exp,ren]] 
    test_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((448, 448))])

    # test_dataset = MyDataset(data_dir=data_dir, transform=test_transform)
    # print('Train Dataset have [%d] samples' % len(test_dataset))
    # test_dataloder = DataLoader(dataset=test_dataset,
    #                              batch_size=batch_size,
    #                              shuffle=True,
    #                              num_workers=0)
    
    # for batch_index, (test_ref, test_exp, test_ren) in enumerate(test_dataloder):
    #     # forward
    #     feature, rendered= model(test_ref, test_exp)

    #     for i in range(batch_size):
    #         pre_ren = rendered[i, :, :, :]
    #         pre_ren = pre_ren.squeeze(dim=0)
    #         img_np = pre_ren.detach().numpy()
    #         maxValue = img_np.max()
    #         img_np = img_np * 255 / maxValue 
    #         mat = np.uint8(img_np)  # float32-->uint8
    #         mat = mat.transpose(1,2,0) 
    #         cv2.imwrite('./run/test/dtn_resnet50/{}-{}'.format(batch_size, i), mat)

    ref_dir = os.path.join(data_dir, 'ref')
    exp_dir = os.path.join(data_dir, 'exp')

    testbar = tqdm(os.listdir(ref_dir), desc='test all test image')
    for ref_test_file in testbar:
        ref_test_path = os.path.join(ref_dir, ref_test_file)
        ref = cv2.imread(ref_test_path) # numpy.ndarray (h,w,c) BGR [0,255]

        prop = ref_test_file.split('-')
        for test_color in test_exp_rendered_color:
            exp_test_file = prop[0] + '-' + test_color[0] + '-' + prop[2]
            exp_test_path = os.path.join(exp_dir, exp_test_file)

            exp = cv2.imread(exp_test_path)

            ref_tensor = test_transform(ref).unsqueeze(dim=0)
            exp_tensor = test_transform(exp).unsqueeze(dim=0)

            # forward
            feature, rendered= model(ref_tensor, exp_tensor)

            feature1 = feature[:, 0 : 3, :, :].squeeze(dim=0)
            feature2 = feature[:, 3 : 6, :, :].squeeze(dim=0)
            feature3 = feature[:, 6 : 9, :, :].squeeze(dim=0)
            feature4 = feature[:, 9 : 12, :, :].squeeze(dim=0)

            # for i, img in enumerate([rendered, feature1, feature2, feature3, feature4]):
            rendered = rendered.squeeze(dim=0)
            img_np = rendered.detach().numpy()
            maxValue = img_np.max()
            img_np = img_np * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
            mat = np.uint8(img_np)  # float32-->uint8
            # print('mat_shape:',mat.shape)  #mat_shape: (3, 982, 814)
            mat = mat.transpose(1,2,0)  # mat_shape: (982, 814，3)
            # mat = cv2.cvtColor(mat,cv2.COLOR_RGB2BGR)
            # cv2.imshow("img",mat)
            cv2.imwrite('./run/test/dtn_resnet50/{}'.format(exp_test_file), mat)
            # cv2.waitKey(0)


