import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from collections import namedtuple


##################################  Original Arcface Model #############################################################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(nn.Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      nn.BatchNorm2d(64), 
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(512 * 7 * 7, 512),
                                       nn.BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

##################################  MobileFaceNet #############################################################
    
class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.model = nn.Sequential(
            Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1)),
            Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64),
            Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
            Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
            Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
            Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)),
            Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        )
        self.flatten = Flatten()
        self.linear = nn.Sequential(
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
    
    def forward(self, x):
        out = self.model(x)
        out = self.linear(self.flatten(out))
        return l2_norm(out)

##################################  Arcface head #############################################################

class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label, age=None):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

    def forward_arccos(self, embbedings, label, age=None):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output = torch.arccos(cos_theta[idx_, label])
        return output

    def forward_original_positive(self, child_embbedings, adult_embbedings, label, age=None):
        # weights norm
        nB = len(child_embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        child_theta = torch.mm(child_embbedings, kernel_norm)
        adult_theta = torch.mm(adult_embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        child_theta = child_theta.clamp(-1, 1)  # for numerical stability
        child_theta_2 = torch.pow(child_theta, 2)

        adult_theta = adult_theta.clamp(-1, 1)  # for numerical stability
        adult_theta_2 = torch.pow(adult_theta, 2)

        child_sin_theta_2 = 1 - child_theta_2
        child_sin_theta = torch.sqrt(child_sin_theta_2)

        adult_sin_theta_2 = 1 - adult_theta_2
        adult_sin_theta = torch.sqrt(adult_sin_theta_2)

        cos_total = child_theta * adult_theta + child_sin_theta * adult_sin_theta
        child_loss = 1 - cos_total
        child_loss = child_loss.mean()
        return child_loss


    # Jooyeol's code
    def get_angle(self, embeddings):
        # Get angles between embeddings and labels
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embeddings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # XXX Cannot understand why this is needed
    
        return cos_theta


##################################  Arcface head with minus margin #############################################################

class ArcfaceMinus(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, minus_m=0.5):
        super(ArcfaceMinus, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.minus_m = minus_m
        print(self.minus_m)
        self.minus_cos_m = math.cos(self.minus_m)
        self.minus_sin_m = math.cos(self.sin_m)


    def forward(self, embbedings, label, age=None):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)

        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        ''' can ignore this code'''
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        ''' can ignore this code'''

        ''' minus margin for child negatives '''
        # cos(theta-m)
        child_idx = torch.where(age == 0)[0]
        if len(child_idx) > 0:
            cos_theta_minus = (cos_theta * self.minus_cos_m + sin_theta * self.minus_sin_m)
            output[child_idx] = cos_theta_minus[child_idx]
        # original Arcface
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

    # Jooyeol's code
    def get_angle(self, embeddings):
        # Get angles between embeddings and labels
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # XXX Cannot understand why this is needed

        return cos_theta

########################################  Cosface head (Real) ####################################

class CosineMarginProduct(nn.Module):
    def __init__(self, embedding_size=512, classnum=10575, scale=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = embedding_size
        self.out_feature = classnum
        self.s = scale
        self.m = m
        self.kernel = nn.Parameter(torch.Tensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.kernel)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label, age=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.kernel))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output

##################################  LDAM head #############################################################

class LDAMLoss(nn.Module):
    def __init__(self, embedding_size=512, classnum=51332, max_m=1.0, s=64, cls_num_list=[]):
        super(LDAMLoss, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.s = s
        print(f'LDAM loss with max_m: {max_m}, scale: {self.s}')
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # child / adult: 2 classes
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list

        assert s > 0
        self.s = s

    def forward(self,embbedings, label, age):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        margin = self.m_list[age]
        margin = margin.unsqueeze(1).expand(cos_theta.shape)

        phi = cos_theta - margin
        label = label.view(-1)  # size=(B,1)
        index = torch.arange(0, label.shape[0], dtype=torch.long)
        output = cos_theta * 1.0
        output[index, index] = phi[index, label]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

############################################### SphereFace Head ########################################################
class SphereMarginProduct(nn.Module):
    def __init__(self, embedding_size, classnum, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0):
        super(SphereMarginProduct, self).__init__()
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_feature = embedding_size
        self.out_feature = classnum
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.kernel = nn.Parameter(torch.Tensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.kernel)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # duplication formula
        self.margin_formula = [
            lambda x : x ** 0,
            lambda x : x ** 1,
            lambda x : 2 * x ** 2 - 1,
            lambda x : 4 * x ** 3 - 3 * x,
            lambda x : 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x : 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label, age=None):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.kernel))
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        # cos_theta = cos_theta(-1, 1)

        cos_m_theta = self.margin_formula[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(input, 2, 1)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output

######################################## Curricular head #####################################################

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label, age=None):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

################################## Grad CAM ################################################

class BackboneMaruta(nn.Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(BackboneMaruta, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      nn.BatchNorm2d(64), 
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(512 * 7 * 7, 512),
                                       nn.BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.gradient = None
        self.gradcam = 11
        # XXX : Change here to change the grad cam layer
        # Good layers 9, 10, 11

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    # x is image
    def get_activations(self, x):
        # [3, 4, 14, 3] bottleneck layers
        lower_body = self.body[:self.gradcam]
        return lower_body(self.input_layer(x))
    
    def forward(self,x):
        x = self.input_layer(x)
        # x = self.body(x)
        lower_body = self.body[:self.gradcam]
        upper_body = self.body[self.gradcam:]
        x = lower_body(x)
        # import pdb; pdb.set_trace()
        h = x.register_hook(self.activations_hook)
        x = upper_body(x)
        x = self.output_layer(x)
        return l2_norm(x)

################################## MVAM / MVArc ################################################
class FC(nn.Module):
    def __init__(self, in_feature=128, out_feature=10572, s=32.0, m=0.50, t=0.2, easy_margin=False, fc_type='MV-AM'):
        super(FC, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.t = t
        self.kernel = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.fc_type = fc_type
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label, ages=None):
        # cos(theta)
        cos_theta = F.linear(F.normalize(x), F.normalize(self.kernel))
        kernel_norm = F.normalize(self.kernel, dim=0)
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score

        if self.fc_type == 'Arc':  # arcface:
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
        elif self.fc_type == 'MV-AM':
            mask = cos_theta > gt - self.m

            hard_vector = cos_theta[mask]

            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.m, gt)
            else:
                final_gt = gt - self.m
        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)

        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.s
        return cos_theta

################################## Broadface ################################################
class BroadFaceArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_factor=64.0,
        margin=0.50,
        queue_size=32000,
        compensate=True,
    ):
        super(BroadFaceArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.margin = margin
        self.scale_factor = scale_factor

        self.kernel = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.kernel)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        feature_mb = torch.zeros(0, in_features)
        label_mb = torch.zeros(0, dtype=torch.int64)
        proxy_mb = torch.zeros(0, in_features)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)
        self.register_buffer("proxy_mb", proxy_mb)

        self.queue_size = queue_size
        self.compensate = compensate

    def update(self, input, label):
        self.feature_mb = torch.cat([self.feature_mb, input.data], dim=0)
        self.label_mb = torch.cat([self.label_mb, label.data], dim=0)
        self.proxy_mb = torch.cat(
            [self.proxy_mb, self.kernel.data[label].clone()], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_mb = self.proxy_mb[over_size:]

        assert (
            self.feature_mb.shape[0] == self.label_mb.shape[0] == self.proxy_mb.shape[0]
        )

    def compute_arcface(self, x, y, w):
        cosine = F.linear(F.normalize(x), F.normalize(w))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        ce_loss = self.criterion(logit, y)
        return ce_loss.mean()

    def forward(self, input, label, ages=None):
        # input is not l2 normalized
        weight_now = self.kernel.data[self.label_mb]
        delta_weight = weight_now - self.proxy_mb

        if self.compensate:
            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / self.proxy_mb.norm(p=2, dim=1, keepdim=True)
                )
                * delta_weight
            )
        else:
            update_feature_mb = self.feature_mb

        large_input = torch.cat([update_feature_mb, input.data], dim=0)
        large_label = torch.cat([self.label_mb, label], dim=0)

        batch_loss = self.compute_arcface(input, label, self.kernel.data)
        broad_loss = self.compute_arcface(large_input, large_label, self.kernel)
        self.update(input, label)

        return batch_loss + broad_loss


################################## Sphereface ################################################

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='sphereface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        # self.kernel = nn.Linear(in_features, out_features, bias=False)
        self.kernel = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.eps = eps

    def forward(self, x, labels, ages=None):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        # kernel = F.normalize(self.kernel, p=2, dim=1)
        # x = F.normalize(x, p=2, dim=1)
        # wf = kernel(x)
        wf = F.linear(F.normalize(x), F.normalize(self.kernel))

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)