
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from mymodels.basetrainer import  BaseTrainer

import math
from mymodels import TRAINER
from myutils.loss import CosLoss

from myutils.loss import   FocalLoss,BinaryFocalLoss

from mymodels.modules.RD_resnet  import resnet18, resnet34, resnet50, wide_resnet50_2,Bottle_Conv

from mymodels.modules.RD_de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50,Bottleneck
from mymodels.modules.RD_de_resnet import  ResNet as DeResNet
import torch.fft

import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.modules.RD_resnet import conv3x3,conv1x1,Type,Union,BasicBlock,Optional,Callable,Tensor,AttnBottleneck

class MoEF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_experts=3):
        super(MoEF, self).__init__()
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.experts = nn.Parameter(torch.randn(num_experts, out_channels, in_channels * 2, kernel_size, kernel_size))
        self.gate = nn.Linear(in_channels * 2, num_experts)

    def forward(self, q, y):
        batch_size, in_channels, height, width = q.size()
        y_context = y.mean([2, 3])
        context = torch.cat([y_context, y_context], dim=1)

        # Calculate gate values based on the concatenated context vector
        weights = self.gate(context)
        weights = torch.softmax(weights, dim=1)

        # Aggregate the weights from the experts
        aggregated_weight = torch.einsum('bi,ijklm->bijklm', weights, self.experts)

        # Concatenate x and y along the channel dimension
        xy = torch.cat([q, q], dim=1)

        # Apply conditional convolution using the aggregated weights
        out = torch.zeros(batch_size, self.out_channels, height, width).to(q.device)
        for i in range(batch_size):
            weight = aggregated_weight[i].view(self.num_experts, self.out_channels, self.in_channels * 2, self.kernel_size, self.kernel_size)
            weight = weight.sum(dim=0)  # Sum along the expert dimension to get the final weight for this sample
            out[i] = F.conv2d(xy[i].unsqueeze(0), weight=weight, stride=1, padding=self.kernel_size // 2)

        return out


class Separation_Module(nn.Module):
    expansion: int = 4
    def __init__(
            self,
            inplanes: int,
            outplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            attention: bool = True,
    ) -> None:
        super(Separation_Module, self).__init__()
        self.attention = attention
        # print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.cbam = GLEAM([int(planes * self.expansion/4),
        #                   int(planes * self.expansion//2),
        #                   planes * self.expansion], 16)
        self.downsample = downsample
        self.stride = stride
        self.out = conv3x3(planes * self.expansion,outplanes)

    def forward(self, x: Tensor) -> Tensor:
        # if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if out.shape[1] == identity.shape[1]:
            out += identity
        out = self.out(out)
        out = self.relu(out)

        return out

#
class Light_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Fusion, self).__init__()
        # self.conv = Bottle_Conv(in_channels,out_channels,kernel_size=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )

    def forward(self, p, i, d):
        att = torch.sigmoid(self.conv_d(d))
        res = self.conv(att*p+i*(1-att))
        return res

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Decoder_ResNet(DeResNet):

    def __init__(
            self,
            **kwargs
    ) -> None:
        super(Decoder_ResNet, self).__init__(
            **kwargs
            # 其他参数未提供，使用默认值
        )
        # del_resnet,_ = wide_resnet50_2(pretrained=True)
        # self.layer3.load_state_dict(del_resnet.layer1.state_dict())
        self.decouple3 = Separation_Module(256,512,64)
        self.decouple2 =  Separation_Module(512,1024,128)
        self.decouple1 = Separation_Module(1024,2048,256)
        self.out1 = nn.Sequential(
            Bottle_Conv(1024, 1024),
            Bottle_Conv(1024, 1024),
        )
        self.out2 = nn.Sequential(
            Bottle_Conv(512, 512),
            Bottle_Conv(512, 512),
        )
        self.out3 = nn.Sequential(
            Bottle_Conv(256, 256),
            Bottle_Conv(256, 256),
        )
        self.cat2 = Light_Fusion(1024, 1024)
        self.cat3 = Light_Fusion(512, 512)


    def _forward_impl(self, x: Tensor, student_out=None) -> Tensor:
        feature_a = self.layer1(x)  # 2048*8*8->1024*16*16
        out_a = self.decouple1(feature_a)
        input_b = self.cat2(*out_a.chunk(2,dim=1), feature_a)

        feature_b = self.layer2(input_b + student_out[2]) if student_out is not None else self.layer2(
            input_b)  # 1024*16*16->512*32*32 #
        out_b= self.decouple2(feature_b)
        input_c = self.cat3(*out_b.chunk(2,dim=1), feature_b)

        feature_c = self.layer3(input_c + student_out[1]) if student_out is not None else self.layer3(
            input_c)  # 1024*32*32->256*64*64 #
        out_c= self.decouple3(feature_c)

        return [out_c, out_b, out_a]

    def forward(self, x: Tensor, student_out=None) -> Tensor:
        return self._forward_impl(x, student_out)
    def forward(self, x: Tensor,student_out=None) -> Tensor:
        return self._forward_impl(x,student_out)

class MRM(nn.Module):
    def __init__(self,
                 block,
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 num_experts=3
                 ):
        super(MRM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        reduce_scale = 1

        self.expand_dim =   Bottle_Conv(64 * block.expansion, 256 * block.expansion, 1,reduce_scale=reduce_scale)
        self.unified_dim_layer3=  nn.Sequential(Bottle_Conv(1024,256,reduce_scale=reduce_scale),
                                             Bottle_Conv(1024, 256,reduce_scale=reduce_scale),
                                             Bottle_Conv(1024, 256,reduce_scale=reduce_scale))
        self.unified_dim_layer2=  nn.Sequential(Bottle_Conv(512,256,stride=2,reduce_scale=reduce_scale),
                                             Bottle_Conv(512, 256,stride=2,reduce_scale=reduce_scale),
                                             Bottle_Conv(512, 256,stride=2,reduce_scale=reduce_scale))
        self.unified_dim_layer1=  nn.Sequential(Bottle_Conv(256,256,stride=4,reduce_scale=reduce_scale),
                                             Bottle_Conv(256, 256,stride=4,reduce_scale=reduce_scale),
                                             Bottle_Conv(256, 256,stride=4,reduce_scale=reduce_scale))

        self.attentions = nn.Sequential(*[MoEF(256,256,num_experts=num_experts) for i in range(6)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion//2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def cross_attention(self,q,rgb,depth,rgb_attention,depth_attention):
        N,C,H,W = q.shape
        res = q +rgb_attention(q,rgb)+depth_attention(q,depth)
        return res
    def _forward_impl(self,rgb_inputs,depth_inputs,student_out) -> Tensor:
        layer3 = [self.unified_dim_layer3[0](student_out[2]),self.unified_dim_layer3[1](rgb_inputs[2]),self.unified_dim_layer3[2](depth_inputs[2])]
        layer2 = [self.unified_dim_layer2[0](student_out[1]),self.unified_dim_layer2[1](rgb_inputs[1]),self.unified_dim_layer2[2](depth_inputs[1])]
        layer1 = [self.unified_dim_layer1[0](student_out[0]),self.unified_dim_layer1[1](rgb_inputs[0]),self.unified_dim_layer1[2](depth_inputs[0])]



        q = self.cross_attention(*layer3,self.attentions[4],self.attentions[5])
        layer2[0] = q+layer2[0]#q + stu layer2
        q = self.cross_attention(*layer2,self.attentions[2],self.attentions[3])
        layer1[0] = q+layer1[0]#q + stu layer2
        q = self.cross_attention(*layer1,self.attentions[0],self.attentions[1])

        feature = self.expand_dim(q)
        output = self.bn_layer(feature)

        return torch.tensor(0.),output.contiguous()

    def forward(self,rgb_inputs,depth_inputs,student_out) -> Tensor:
        return self._forward_impl(rgb_inputs,depth_inputs,student_out)
@TRAINER.register
class DecoupleMAD(BaseTrainer):
    def __init__(self,args,num_experts = 3):
        super(DecoupleMAD, self).__init__()

        self.encoder,_ = wide_resnet50_2(pretrained=True)
        self.encoder = self.encoder.cuda()
        self.encoder.eval()
        self.bn = MRM(AttnBottleneck,1,num_experts=num_experts).cuda()

        self.stu_encoder = nn.Sequential(nn.Conv2d(6,3,kernel_size=3,padding=1),
                                         # nn.BatchNorm2d(3),
                                         # nn.ReLU(),
                                         wide_resnet50_2(pretrained=True)[0]
                                         ).cuda()
        self.decoder  =  Decoder_ResNet(block = Bottleneck,layers=[3,4,6,3],width_per_group=128).cuda()
        self.loss_focal = BinaryFocalLoss(reduce=False)
        self.loss_smL1 = nn.SmoothL1Loss(reduction='none')
        # self.model_seg_aux = SmallDiscriminativeSubNetwork(in_channels=12, out_channels=2)
        # self.model_seg_aux.cuda()
        self.loss_cos = CosLoss(avg=False)

    def loss_fucntion(self,a, b,weight=None,reduction="mean"):

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            temp2 = 1 - cos_loss(a[item].reshape(a[item].shape[0], -1),b[item].reshape(b[item].shape[0], -1))
            temp = temp2.mean()#+temp1.mean()
            if reduction=="mean":
                loss += torch.mean(temp)
            else:loss+=temp
        return loss

    def train_step(self, batch, **kwargs):
        self.decoder.train()
        self.bn.train()
        self.encoder.eval()
        image = batch['RGB']

        depth_image = batch['Depth']
        loss,amap = self.forward_step(image,depth_image,True)
        return loss, {"Total_Loss": loss.item(),
                      }

    def forward(self, batch, **kwargs):
        return self.eval_step(batch, **kwargs)
    def eval_step(self, batch, **kwargs):
        self.decoder.train()
        self.bn.train()
        self.encoder.eval()
        image = batch['RGB']

        depth_image = batch['Depth']
        _,amap  = self.forward_step(image,depth_image,False)

        return image, amap
    def cos_sim(self,x,rec_x,size=(256,256)):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        rec_x = F.interpolate(rec_x, size=size, mode='bilinear', align_corners=True)
        sim = 1 - F.cosine_similarity(x, rec_x, dim=1)
        return sim
    def forward_step(self, image,depth_image,train=False):
        stu_input = torch.cat([image,depth_image],1)

        rgb_inputs = self.encoder(image)
        # depth_inputs = self.encoder(depth_image.repeat(1,3,1,1))
        depth_inputs = self.encoder(depth_image)
        student_out = self.stu_encoder(stu_input)
        _,bn_out = self.bn(rgb_inputs,depth_inputs,student_out)
        outputs = self.decoder(bn_out,student_out)

        amaps = 0
        loss = 0
        for i,(p, q) in enumerate(zip(rgb_inputs, depth_inputs)):

            rec_p,rec_q = outputs[i].chunk(2,dim=1)
            p,q = F.relu(p),F.relu(q)
            rec_p,rec_q = F.relu(rec_p),F.relu(rec_q)
            loss = loss + self.loss_fucntion([p,q], [ rec_p,rec_q], None)

            if not train:
                amap = torch.sqrt((self.cos_sim(p,rec_p)**2+self.cos_sim(q,rec_q)**2))/2
                amaps+=amap
        return loss,amaps

    def get_models(self):
        return ("self.decoder, self.bn,  self.stu_encoder".split(","),
                 self.decoder, self.bn,  self.stu_encoder)
