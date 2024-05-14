from torch import nn
from torch_optimizer import DiffGrad
from torchvision import models
import torch

class Se(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(Se, self).__init__()
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=self.fc(out.view(out.size(0),-1))
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x  

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, context=None):
        q = self.q(x)
        if context==None: context = x
        k = self.k(context)
        v = self.v(context)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        x = torch.bmm(v,w_)     # b, c,hw (hw of q) x[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        x = x.reshape(b,c,h,w)

        x = self.proj_out(x)

        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn1 = CrossAttention(in_channels)  # is a self-attention
        self.ff = torch.nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.attn2 = CrossAttention(in_channels)  # is self-attn if context is none
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.norm3 = nn.BatchNorm2d(in_channels)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class ThreeNet(nn.Module):
    def __init__(self, lr=1e-4):
        super(ThreeNet,self).__init__()
        resnet18Pretrained1 = models.resnet18(pretrained=True)
        resnet18Pretrained2 = models.resnet18(pretrained=True)
        resnet18Pretrained3 = models.resnet18(pretrained=True)

        ## color branch
        self.b1_1 = nn.Sequential(*list(resnet18Pretrained1.children())[:5])
    
        self.b1_2 = list(resnet18Pretrained1.children())[5]
    
        self.b1_3 = list(resnet18Pretrained1.children())[6]
        self.attn1_3 = AttnBlock(256)
        self.se1_3 = Se(256)

        self.b1_4 = list(resnet18Pretrained1.children())[7]

        ## gray branch
        self.b2_1 = nn.Sequential(*list(resnet18Pretrained2.children())[:5])
    
        self.b2_2 = list(resnet18Pretrained2.children())[5]
    
        self.b2_3 = list(resnet18Pretrained2.children())[6]
        self.attn2_3 = AttnBlock(256)
        self.se2_3 = Se(256)

        self.b2_4 = list(resnet18Pretrained2.children())[7]

        ## sketch branch
        self.b3_1 = nn.Sequential(*list(resnet18Pretrained3.children())[:5])
    
        self.b3_2 = list(resnet18Pretrained3.children())[5]
    
        self.b3_3 = list(resnet18Pretrained3.children())[6]
        self.attn3_3 = AttnBlock(256)
        self.se3_3 = Se(256)

        self.b3_4 = list(resnet18Pretrained3.children())[7]

        self.se = Se(512)

        self.transformer1 = BasicTransformerBlock(512)
        self.transformer2 = BasicTransformerBlock(512)

        self.last_block = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.opt = DiffGrad(self.parameters(), lr=lr, betas=(0.5, 0.9))
        self.sigmoid = nn.Sigmoid()
        self.cuda()

    def forward(self, x1, x2, x3):
        
        x1 = self.b1_1(x1)
        x1 = self.b1_2(x1)
        x1 = self.b1_3(x1)
        x1 = self.attn1_3(x1)
        x1 = self.se1_3(x1)
        x1 = self.b1_4(x1)

        x2 = self.b2_1(x2)
        x2 = self.b2_2(x2)
        x2 = self.b2_3(x2)
        x2 = self.attn2_3(x2)
        x2 = self.se2_3(x2)
        x2 = self.b2_4(x2)

        x3 = self.b3_1(x3)
        x3 = self.b3_2(x3)
        x3 = self.b3_3(x3)
        x3 = self.attn3_3(x3)
        x3 = self.se3_3(x3)
        x3 = self.b3_4(x3)

        x1 = self.transformer1(x1, context=x2)
        score = self.transformer2(x3, context=x1)
        score = self.se(score)

        score = self.last_block(score)
        score = self.avg_pool(score)

        score = self.sigmoid(score)

        return score.squeeze()