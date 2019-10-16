import torch
from PIL import Image
import os
import numpy
import lenet5
import re

label_str = ['presion', 'car', 'bus', 'truck', 
             'microbus', 'taxi', 'bicycle', 'electrombile', 
             'motorcycle', 'bigtruck', 'trailer', 'tractor']

path_net = './car_classifier_net.pth'
path_image = 'images_labels/'
path_output = 'images_preds/'
net = lenet5.Net()
net.load_state_dict(torch.load(path_net))

# clear the path
for filename in os.listdir(path_output):
    os.remove(path_output + filename)

count = 0
for filename in os.listdir(path_image):
    count = count + 1
    # get label from filename
    num = re.findall(r"\d+\.?\d*",filename)
    real = int(num[1])
    im = Image.open(path_image + filename)
    im = im.resize((32,32))
    imts = torch.Tensor(numpy.transpose(numpy.array(im).reshape(32, 32, 3), (2, 0, 1)))
    data = torch.zeros(1,3,32,32)
    data[0] = imts
    pred = net(data)
    pred_str = label_str[int(numpy.argmax(pred.detach().numpy(), axis=1))]
    real_str = label_str[real]
    name = path_output + str(count) + '_pre[' + pred_str + ']_real[' + real_str + '].jpg'
    im.save(name)
    print('save image : ' + name)
print('finish testing')