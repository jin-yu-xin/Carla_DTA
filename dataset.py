import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from torch.utils.data import Dataset

# ['17,37,103', '75,86,173', '180,42,42', '0,0,0', '137,0,0'] # r,g,b
exp_colors = [['001', [103, 37, 17]], 
             ['002', [173, 86, 75]], 
             ['003', [42, 42, 180]], 
             ['004', [0, 0, 0]], 
             ['005', [0, 0, 137]]] # [index, [b, g, r]]
             

class MyDataset(Dataset):
    def __init__(self, data_dir, exp_colors, transform=None):
        self.data_info = self.get_data_info(data_dir, exp_colors)  # list [[ref,exp,ren],[ref,exp,ren]]
        self.transform = transform
    
    def __getitem__(self, index):
        ref_path, exp_path, ren_path = self.data_info[index]
        ref = cv2.imread(ref_path) # numpy.ndarray (h,w,c) BGR [0,255]  image_size: 1080x720
        exp = cv2.imread(exp_path)
        ren = cv2.imread(ren_path)
        if self.transform is not None:
            ref = self.transform(ref)
            exp = self.transform(exp)
            ren = self.transform(ren)

        return ref, exp, ren  # tensor (c,h,w) [0.0, 1.0]
    
    def __len__(self):
        return len(self.data_info)

    @staticmethod    
    def get_data_info(data_dir, exp_colors):
        pairs = []
        ref_dir = os.path.join(data_dir, 'ref')
        exp_dir = os.path.join(data_dir, 'exp')
        ren_dir = os.path.join(data_dir, 'ren')
        if os.path.exists(ref_dir) and os.path.exists(exp_dir) and os.path.exists(ren_dir):
            ref_file_list = os.listdir(ref_dir)
            if len(ref_file_list) == 0:
                print('there is no ref file!!!')
            else:
                for ref_file in ref_file_list:
                    ref_file_path = os.path.join(ref_dir, ref_file)
                    prop = ref_file.split('-')
                    for color in exp_colors:
                        pair = []
                        # get ref
                        pair.append(ref_file_path)
                        # get exp
                        exp_file = prop[0] + '-' + color[0] + '-' + prop[2]
                        exp_file_path = os.path.join(exp_dir, exp_file)
                        pair.append(exp_file_path)
                        # get ren
                        ren_file =  prop[0] + '-' + color[0] + '-' + prop[2]
                        ren_file_path = os.path.join(ren_dir, ren_file)
                        pair.append(ren_file_path)

                        pairs.append(pair)
        else:
            print('there is no ref or exp or ren folder!!!')
        return pairs