import os
import re
from PIL import Image
import torch
import torch.utils.data as data 
import numpy



class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        
        images_labels_path = 'images_labels/'
        count = 0
        for filename in os.listdir(images_labels_path):
            count = count + 1
        
        x = torch.zeros((count))
        y = torch.zeros((count, 3, 32, 32))
        ii = 0
        for filename in os.listdir(images_labels_path):
            num = re.findall(r"\d+\.?\d*",filename)
            x[ii] = int(num[1])
            im = Image.open(images_labels_path + filename)
            im = im.resize((32,32))
            imts = torch.Tensor(numpy.transpose(numpy.array(im).reshape(32, 32, 3), (2, 0, 1)))
            y[ii] = imts
            ii = ii + 1
        self.x = x.long()
        self.y = y
        self.len = count

    def __getitem__(self, item):
        return self.y[item], self.x[item]

    def __len__(self):
        return self.len

def load_data():
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    dataset = Dataset()
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return data_loader
