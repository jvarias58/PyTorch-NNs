import torch
from torch import nn

right = 0
for i in range(1000):
    img = test_data[i][0].unsqueeze(dim=0)
    label = test_data[i][1]
    a = model(img)
    a = torch.argmax(a,dim=1)
    if (a.item() == label):
        right +=1
print(right)