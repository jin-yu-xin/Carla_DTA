import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from dtn import DTN


def init_adversarial_pattern(h=720, w=1080):
    adv_np = np.random.random((h, w, 3))  # [0, 1) [Height, Width, Channels]
    adv_image = (adv_np * 255).astype('uint8')
    # print(adv_image.shape)
    # cv2.imshow('texture', adv_image)
    cv2.imwrite('./carla-datasets/datasets/test/texture/adv_image.png', adv_image)
    # cv2.waitKey(0)

    adv = torch.from_numpy(adv_np).permute(2, 0, 1)
    print(adv.shape)


def render_one_image(file='000-000-005.png', texture_file='adv_image.png'):
    mask_dir = './carla-datasets/datasets/test/mask'
    ref_dir = './carla-datasets/datasets/test/ref'
    texture_dir = './carla-datasets/datasets/test/texture'

    mask_file_path = os.path.join(mask_dir, file)
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)  # 8-bit one channel
    ref_test_path = os.path.join(ref_dir, file)
    ref = cv2.imread(ref_test_path) # numpy.ndarray (h,w,c) BGR [0,255]

    # make exp
    texture_path = os.path.join(texture_dir, texture_file)
    if os.path.exists(texture_path):
        color_texture = cv2.imread(texture_path)
        exp = cv2.bitwise_and(color_texture, color_texture, mask=mask)
    
    transform = transforms.Compose([transforms.ToTensor(),   # [0,255]->[0,1] (h,w,c)->(c,h,w)
                                    transforms.Resize((448, 448))])
    
    ref_tensor = transform(ref).unsqueeze(dim=0)
    exp_tensor = transform(exp).unsqueeze(dim=0)

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
    mat = mat.transpose(1,2,0)  # mat_shape: (982, 814，3)
    cv2.imwrite('rendered_car.png', mat)



# init_adversarial_pattern()
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = DTN(ae_type='ResNet50')
    model_dict = torch.load('./run/train/weight/dtn_resnet50_NoRandomHorizontalFlip_100.pth')
    model.load_state_dict(model_dict)
    print('load model successfully!!!')
    # init_adversarial_pattern()
    render_one_image(file='000-000-008.png', texture_file='adv_image.png')