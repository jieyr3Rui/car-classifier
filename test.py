import torch
from PIL import Image
import os
import numpy
import lenet5

path_net = 'cifar_net.pth'
path_image = 'images_labels/'
path_output = 'images_preds/'
net = lenet5.Net()
net.load_state_dict(torch.load(path_net))

count = 0
for filename in os.listdir(path_image):
    count = count + 1

    im = Image.open(path_image + filename)
    im = im.resize((32,32))
    imts = torch.Tensor(numpy.transpose(numpy.array(im).reshape(32, 32, 3), (2, 0, 1)))
    data = torch.zeros(1,3,32,32)
    data[0] = imts
    pred = net(data)
    print(numpy.argmax(pred.detach().numpy(), axis=1))
    im.save(path_output + str(count) + '_' + str(numpy.argmax(pred.detach().numpy(), axis=1)) + '.jpg')