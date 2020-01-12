import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, embd_size):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, embd_size)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 
        'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 
        'M', 512, 512, 512, 512, 'M'],
}


class EmbeddingNetVGG(nn.Module):
    def __init__(self, embd_size):
        vgg_name='VGG11'
        super(EmbeddingNetVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class EmbeddingNetRGB(nn.Module):
    def __init__(self, embd_size):
        super(EmbeddingNetRGB, self).__init__()
        # self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), padding=(2,2), stride=(1,1)),
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(1,1)),
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(1,1)),
                                     # nn.Conv2d(32, 64, 5), 
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        # self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
        self.fc = nn.Sequential(nn.Linear(64 * 5 * 5 , 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, embd_size)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)




class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiameseNet_ClassNet(nn.Module):
    def __init__(self, embedding_net, n_classes, embd_size):
        super(SiameseNet_ClassNet, self).__init__()
        self.embedding_net = embedding_net
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(embd_size, n_classes)
        # self.fc2 = nn.Linear(2, n_classes)

    def forward(self, x1, x2, outraw1, outraw2):
        '''
        Outraw is for making embedding outlook
        '''
        if type(outraw1) == type(None):
            outraw1 = self.embedding_net(x1)
        # scores1 = F.log_softmax(self.fc1(output1), dim=-1)
        output1 = self.nonlinear(outraw1)
        scores1 = F.log_softmax(self.fc1(output1), dim=-1)
       
        if type(outraw2) == type(None):
            outraw2= self.embedding_net(x2)
        # scores2 = F.log_softmax(self.fc2(output2), dim=-1)
        output2  = self.nonlinear(outraw2)
        scores2 = F.log_softmax(self.fc1(output2), dim=-1)
        
        return outraw1, outraw2, scores1, scores2

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
