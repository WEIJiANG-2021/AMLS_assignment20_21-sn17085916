import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

celeb_label = {'-1': 0, '1': 1}


class CELEBAData(Dataset):
    def __init__(self, data_dir, transform=None):

        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        root = os.path.join(data_dir, 'img')
        csv_path = os.path.join(data_dir, 'labels.csv')

        f = pd.read_csv(csv_path, sep='\t')
        for i, row in f.iterrows():
            img_name, label = row['img_name'], row['smiling']
            path_img = os.path.join(root, img_name)
            l = celeb_label[str(label)]
            data_info.append((path_img, int(l)))
        return data_info
