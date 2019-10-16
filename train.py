import lenet5
import load_data
import torch
import torch.nn as nn

net = lenet5.Net().cuda()

d = load_data.load_data()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_function = torch.nn.CrossEntropyLoss()
losses = []
for epoch in range(500):
    running_loss = 0.0
    for ii, train_data in enumerate(d):
        data = train_data[0].cuda()
        label = train_data[1].cuda()
        
        pred = net(data)
        loss = loss_function(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
    loss_step = running_loss / (ii + 1)
    
    losses.append(loss_step)
    print('epoch: %4d, loss = %.3f' % (epoch, float(loss_step)))

PATH = './car_classifier_net.pth'
torch.save(net.state_dict(), PATH)