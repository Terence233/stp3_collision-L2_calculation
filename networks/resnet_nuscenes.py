from torch import Tensor
from torch import nn
import torch
from torchvision import models

class FCModel(nn.Module):
    def __init__(self, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()

        self.fc = nn.Linear(in_features=8, out_features=1000)
        self.relu = nn.ReLU(inplace=True)

        self.final_fc = nn.Linear(in_features=1000, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.relu(x)
        x = self.final_fc(x)
        return x

class ResNetModel(nn.Module):
    def __init__(self, type=50, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()
        if pretrained:
            weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
        else:
            weights = None
        
        if type == 50:
            self.model = models.resnet50(weights = weights)
        elif type == 101:
            self.model = models.resnet101(weights = weights)
        else:
            self.model = models.resnet152(weights = weights)

        in_channels = input_n * 3
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.fc = nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1, 2)
        x = self.model(x)
        return x
    
class ResNetModel_R(nn.Module): # rotation
    def __init__(self, type=50, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()

        self.input_n = input_n
        self.out_features = out_features

        if pretrained:
            weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
        else:
            weights = None
        
        if type == 50:
            self.model = models.resnet50(weights = weights)
        elif type == 101:
            self.model = models.resnet101(weights = weights)
        else:
            self.model = models.resnet152(weights = weights)

        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=256)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_features=1000 * input_n, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(0, 1)

        x = self.model(x)

        x = self.relu(x)

        x = x.view(-1, 1000 * self.input_n)

        x = self.fc(x)

        return x
    
class ResNetModel_C(nn.Module): # Coordinate 
    def __init__(self, type=50, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()

        self.input_n = input_n
        self.out_features = out_features

        if pretrained:
            weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
        else:
            weights = None
        
        if type == 50:
            self.model = models.resnet50(weights = weights)
        elif type == 101:
            self.model = models.resnet101(weights = weights)
        else:
            self.model = models.resnet152(weights = weights)

        self.model.conv1 = nn.Conv2d(3*input_n, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc = nn.Linear(in_features=2048, out_features=256)
        self.coor_fc = nn.Linear(in_features=2*input_n-2, out_features=1000)
        self.relu = nn.ReLU(inplace=True)

        self.final_fc = nn.Linear(in_features=2*1000, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        x[0] = x[0].flatten(1, 2)

        x[0] = self.model(x[0])
        x[1] = self.coor_fc(x[1]) 

        x = torch.cat((x[0], x[1]), dim=1)
        #x = x[0] * x[1]

        x = self.relu(x)

        x = self.final_fc(x)

        return x


def resnet_model(type=50, pretrained=True, out_features=12):
    if pretrained:
        weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
    else:
        weights = None
    
    if type == 50:
        model = models.resnet50(weights = weights)
    elif type == 101:
        model = models.resnet101(weights = weights)
    else:
        model = models.resnet152(weights = weights)

    model.fc = nn.Linear(in_features=2048, out_features=out_features)
    return model


# # 测试网络
# if __name__ == '__main__':
#     model = ResNetModel_C()
#     loss_function = torch.nn.L1Loss()
#     input = torch.FloatTensor([1,2,3])
#     target = torch.FloatTensor([4,8,10])
#     loss = loss_function(input, target)
#     print(loss.item())