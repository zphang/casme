import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .ext.pytorch_inpainting_with_partial_conv import PConvUNet, PCBActiv


class PConvUNetGEN(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest', infoGAN = False):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            if i == self.layer_size-1:
                if infoGAN:
                    setattr(self, name, PCBActiv(512 + 512 + 4, 512, activ='leaky'))
                else:
                    setattr(self, name, PCBActiv(512 + 512 + 1, 512, activ='leaky'))
            else:
                setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)
        
        self.infoGAN = infoGAN
        self.emb = nn.Embedding(3, 3) 
        self.emb.requires_grad = False
        self.emb.weight.data = torch.eye(3)
        self.upsample = nn.Upsample(scale_factor=2**(8 - layer_size + 1), mode='nearest') # TODO: fix, rather than using magic number 8
        self.num_labels = 3 # TODO: fix

    def forward(self, input, input_mask, labels=None):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')
            if i == self.layer_size:
                # inserts random channel
                random_input = torch.rand(h.size(0),1,h.size(2),h.size(3)).cuda()
                if self.infoGAN:
                    label_channels = self.emb(labels).view(labels.shape[0], self.num_labels, 1, 1)
                    label_channels = self.upsample(label_channels)
                    h = torch.cat([h, h_dict[enc_h_key], random_input, label_channels], dim=1)
                    h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key], torch.ones_like(random_input), torch.ones_like(label_channels)], dim=1)
                else:
                    
                    h = torch.cat([h, h_dict[enc_h_key], random_input], dim=1)
                    h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key], torch.ones_like(random_input)], dim=1)
                
            else:
                h = torch.cat([h, h_dict[enc_h_key]], dim=1)
                h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetShared(ResNet):
    def forward(self, x, return_intermediate=False):
        l = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if return_intermediate:
            return x, l
        else:
            return x


def resnet50shared(pretrained=False, **kwargs):
    model = ResNetShared(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

class Discriminator(nn.Module):
    def __init__(self, input_dim, return_logits=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.return_logits = return_logits

    def forward(self, h, resnet):
        h = resnet.final_bn(h)
        h = resnet.relu(h)
        pooled_h = h.view(h.shape[0], h.shape[1], -1).mean(dim=2)
        # TODO: detach?
        logits = self.fc1(pooled_h)#.detach())
        #print(logits)
        if self.return_logits:
            output = logits
        else:
            output = F.log_softmax(logits, dim=1)
        return output


class Infiller(nn.Module):

    def __init__(self, model_type, input_channels, num_layers=6):
        super().__init__()
        self.model_type = model_type
        self.input_channels = input_channels
        self.num_layers = num_layers
        if model_type == "ciGAN":
            # do I have a mask for each category, 1 indicating salient region?
            pass
        elif model_type =="pconv":
            self.model = PConvUNet(layer_size=num_layers, input_channels=input_channels)
        elif model_type == "pconv_gan":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels)
            #self.infiller = PConvUNet(layer_size=num_layers, input_channels=input_channels)
            #pass
        elif model_type == "pconv_infogan":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels, infoGAN=True)
            #self.infiller = PConvUNet(layer_size=num_layers, input_channels=input_channels)
            #pass
        else:
            raise NotImplementedError()

    def forward(self, x, mask, labels=None):
        if self.model_type == "ciGAN":
            pass
        #elif self.model_type == "pconv":
        elif self.model_type in ["pconv", "pconv_gan"]:
            return self.model(x, mask)
        elif self.model_type == "pconv_infogan":
            return self.model(x, mask, labels)
        #elif self.model_type == "pconv_gan":
        #    pass
        else:
            raise NotImplementedError()


class Masker(nn.Module):

    def __init__(self, in_channels, out_channel,
                 final_upsample_mode='nearest', add_prob_layers=False):
        super().__init__()
        self.add_prob_layers = add_prob_layers

        p_dim = 1 if self.add_prob_layers else 0
        self.conv1x1_1 = self._make_conv1x1_upsampled(in_channels[1] + p_dim, out_channel)
        self.conv1x1_2 = self._make_conv1x1_upsampled(in_channels[2] + p_dim, out_channel, 2)
        self.conv1x1_3 = self._make_conv1x1_upsampled(in_channels[3] + p_dim, out_channel, 4)
        self.conv1x1_4 = self._make_conv1x1_upsampled(in_channels[4] + p_dim, out_channel, 8)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels[0] + 4 * out_channel + p_dim, 1,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode=final_upsample_mode)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @classmethod
    def _make_conv1x1_upsampled(cls, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, l, use_p):
        if self.add_prob_layers:
            l = self.append_p(l, use_p)
        else:
            assert use_p is None

        k = [
            l[0],
            self.conv1x1_1(l[1]),
            self.conv1x1_2(l[2]),
            self.conv1x1_3(l[3]),
            self.conv1x1_4(l[4]),
        ]
        return self.final(torch.cat(k, 1))

    def append_p(self, l, p):
        new_l = []
        for layer in l:
            p_slice = p \
                .expand(1, layer.shape[2], layer.shape[3], -1) \
                .permute(3, 0, 1, 2)
            new_l.append(torch.cat([layer, p_slice], dim=1))
        return new_l

def decoder(**kwargs):
    return Masker([64, 256, 512, 1024, 2048], 64, **kwargs)
