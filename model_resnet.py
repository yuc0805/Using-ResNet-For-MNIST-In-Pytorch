import torch
import torch.nn as  nn
import torch.nn.functional as F

'''
According to the paper https://arxiv.org/abs/1512.03385
We will need a bottleneck layer for Res50+ due to dimension differences
'''

class Bottleneck(nn.Module):
    def __init__(self,in_channels,intermediate_channels,
                 expansion,is_Bottleneck,stride):
        super(Bottleneck,self).__init__()

        self.expansion = expansion #expansion = 4 for Res50+
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        self.out_channels = self.intermediate_channels*self.expansion
        
        self.relu = nn.ReLU()
        
        # Bottleneck Block
        if self.is_Bottleneck:
            #1x1
            self.conv1_1x1 = nn.Conv2d(self.in_channels,self.intermediate_channels,
                                       kernel_size=1,stride=1,padding=0,bias=False)
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            #3x3
            self.conv2_3x3 = nn.Conv2d(self.intermediate_channels,self.intermediate_channels,
                                       kernel_size=3,stride=stride,padding=1,bias=False)
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)
            
            #1x1
            self.conv3_1x1 = nn.Conv2d(self.intermediate_channels,self.out_channels,
                                       kernel_size=1,stride=1,padding=0,bias=False)
            self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        
        # Basic Block (18,34 layer)
        else:
            #3x3
            self.conv1_3x3 = nn.Conv2d(self.in_channels,self.intermediate_channels,
                                       kernel_size=3,stride=stride,padding=1,bias=False)
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            #3x3
            self.conv2_3x3 = nn.Conv2d(self.intermediate_channels,self.intermediate_channels,
                                       kernel_size=3,stride=1,padding=1,bias=False)
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)


        #check if need projection for residual connection
        if self.in_channels ==  self.out_channels:
            self.identity = True
        #project in_channels match out_channels
        else:
            self.identity = False
            projection_layer = []
            projection = nn.Conv2d(self.in_channels, self.out_channels,kernel_size=1,
                      stride=stride, padding=0,bias=False)
            projection_layer.append(projection)
            projection_norm = nn.BatchNorm2d(self.out_channels)
            projection_layer.append(projection_norm)
            
            self.projection = nn.Sequential(*projection_layer)

    def forward(self,x):
        #store input for residual connection
        in_x = x

        if self.is_Bottleneck:
            # 1x1
            x = self.batchnorm1(self.conv1_1x1(x))
            x = self.relu(x)

            # 3x3
            x = self.batchnorm2(self.conv2_3x3(x))
            x = self.relu(x)

            # 1x1
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            x = self.batchnorm1(self.conv1_3x3(x))
            x = self.relu(x)

            x = self.batchnorm2(self.conv2_3x3(x))

        # check for residual connection
        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        x = self.relu(x)

        return x
    
class ResNet(nn.Module):
    def __init__(self, resnet_variant, in_channels,num_classes):
        super(ResNet,self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(in_channels, out_channels=64,
                               kernel_size=7, stride=2,padding=3,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.block1 = self._make_blocks(64,self.channels_list[0], 
                                        self.repeatition_list[0], self.expansion, 
                                        self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0]*self.expansion,
                                        self.channels_list[1], self.repeatition_list[1], 
                                        self.expansion,self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1]*self.expansion,
                                        self.channels_list[2], self.repeatition_list[2], 
                                        self.expansion,self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2]*self.expansion,
                                        self.channels_list[3], self.repeatition_list[3], 
                                        self.expansion,self.is_Bottleneck, stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3]*self.expansion,num_classes)

    def _make_blocks(self,in_channels,intermediate_channels,num_repeat,
                     expansion,is_Bottleneck,stride):
        
        blocks = []
        blocks.append(Bottleneck(in_channels,intermediate_channels,expansion,
                                 is_Bottleneck,stride=stride))
        
        for num in range(1,num_repeat):
            blocks.append(Bottleneck(intermediate_channels*expansion,
                                     intermediate_channels,expansion,is_Bottleneck,stride=1))

        return nn.Sequential(*blocks)
    
    def forward(self,x):
        x = self.batchnorm1(self.conv1(x))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.average_pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

# Some ResNet architecture suggessted by the original paper:
def resnet_18(**kwargs):
    model = ResNet([[64,128,256,512],[2,2,2,2],1,False],**kwargs)
    return model

def resnet_34(**kwargs):
    model = ResNet([[64,128,256,512],[3,4,6,3],1,False],**kwargs)
    return model

def resnet_50(**kwargs):
    model = ResNet([[64,128,256,512],[3,4,6,3],4,True],**kwargs)
    return model

def resnet_101(**kwargs):
    model = ResNet([[64,128,256,512],[3,4,23,3],4,True],**kwargs)
    return model

def resnet_152(**kwargs):
    model = ResNet([[64,128,256,512],[3,8,36,3],4,True],**kwargs)
    return model
            
#Test network
if __name__ == '__main__':
    model = resnet_18(in_channels=1,num_classes=10)
    print(model)



