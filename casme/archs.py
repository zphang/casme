import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .ext.pytorch_inpainting_with_partial_conv import PConvUNet, PCBActiv


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class PConvUNetGEN(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest', infoGAN=False,
                 final_activation=None):
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

        self.final_activation = final_activation

    def forward(self, input, input_mask, labels=None):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        import collections as col
        zstorage = col.OrderedDict()

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key
            # print(h_key, tuple(h_dict[h_key].shape))
            zstorage["A_h_"+h_key] = h_dict[h_key]
            zstorage["A_m_" + h_key] = h_mask_dict[h_key]

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
            zstorage["B_h_{}".format(i)] = h
            zstorage["B_m_{}".format(i)] = h_mask
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
                    # print(i, h.shape)
                    zstorage["B_h_{}".format(i)] = h
                    zstorage["B_m_{}".format(i)] = h_mask
                
            else:
                h = torch.cat([h, h_dict[enc_h_key]], dim=1)
                h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
                # print(i, h.shape)
                zstorage["C_h_{}".format(i)] = h
                zstorage["C_m_{}".format(i)] = h_mask
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            # print(i, h.shape)
            zstorage["D_h_{}".format(i)] = h
            zstorage["D_m_{}".format(i)] = h_mask

        if self.final_activation is None:
            pass
        elif self.final_activation == "tanh":
            h = torch.tanh(h)
        else:
            raise NotImplementedError()

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
        elif model_type == "pconv_infogan":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels, infoGAN=True)
        elif model_type == "pconv_gan2":
            self.model = PConvUNetGEN(layer_size=num_layers, input_channels=input_channels,
                                      final_activation="tanh")
        elif model_type == "none":
            # To stop PyTorch from complaining about not having parameters
            self.model = torch.nn.Linear(1, 1)
        else:
            raise NotImplementedError()

    def forward(self, x, mask, labels=None):
        if self.model_type == "ciGAN":
            pass
        elif self.model_type in ["pconv", "pconv_gan"]:
            return self.model(x, mask)
        elif self.model_type == "pconv_infogan":
            return self.model(x, mask, labels)
        elif self.model_type == "pconv_gan2":
            return self.model(x, mask)
        elif self.model_type == "none":
            return x, mask
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
            Upsample(scale_factor=4, mode=final_upsample_mode),
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
                Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, layers, use_p):
        if self.add_prob_layers:
            assert use_p is not None
            if not isinstance(use_p, torch.Tensor):
                batch_size = layers[0].shape[0]
                device = layers[0].device
                use_p = torch.Tensor([use_p] * batch_size).to(device)

            layers = self.append_p(layers, use_p)
        else:
            assert use_p is None

        k = [
            layers[0],
            self.conv1x1_1(layers[1]),
            self.conv1x1_2(layers[2]),
            self.conv1x1_3(layers[3]),
            self.conv1x1_4(layers[4]),
        ]
        return self.final(torch.cat(k, 1))

    @classmethod
    def append_p(cls, layers, p):
        new_layers = []
        for layer in layers:
            p_slice = p \
                .expand(1, layer.shape[2], layer.shape[3], -1) \
                .permute(3, 0, 1, 2)
            new_layers.append(torch.cat([layer, p_slice], dim=1))
        return new_layers


class InfillerCNN(nn.Module):
    def __init__(self, in_chans, out_chans, intermediate_dim_ls):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.intermediate_dim_ls = intermediate_dim_ls
        self.down_ls = nn.ModuleList()
        self.up_ls = nn.ModuleList()
        self.n_dims = len(intermediate_dim_ls)

        self.down_ls.append(self.get_conv_layer(
            in_chans, intermediate_dim_ls[0],
        ))
        for i in range(0, self.n_dims - 1):
            self.down_ls.append(self.get_conv_layer(
                intermediate_dim_ls[i],
                intermediate_dim_ls[i + 1],
            ))
        self.up_ls.append(self.get_conv_layer(
            intermediate_dim_ls[-1],
            intermediate_dim_ls[-2],
        ))
        for i in list(range(self.n_dims - 2, 0, -1)):
            self.up_ls.append(self.get_conv_layer(
                self.up_ls[-1].out_channels + intermediate_dim_ls[i],
                intermediate_dim_ls[i],
            ))
        self.up_ls.append(self.get_conv_layer(
            intermediate_dim_ls[0] + intermediate_dim_ls[1],
            out_chans,
        ))
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.init_weights()

    def forward(self, masked_x, mask):
        h = torch.cat([masked_x, mask], dim=1)
        down_h_ls = []
        for i in range(self.n_dims):
            h = self.max_pool(self.relu(self.down_ls[i](h)))
            down_h_ls.append(h)
        for i in range(self.n_dims - 1):
            h = self.upsample(self.relu(self.up_ls[i](h)))
            h = torch.cat([h, down_h_ls[-i - 2]], dim=1)
        h = self.upsample(self.up_ls[-1](h))
        return h

    @classmethod
    def get_conv_layer(cls, in_channels, out_channels):
        return nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

    def init_weights(self):
        for conv_layer in list(self.down_ls) + list(self.up_ls):
            torch.nn.init.xavier_uniform_(conv_layer.weight)


class CAInfillerWrapper(nn.Module):
    def __init__(self, iproc):
        super().__init__()
        self.iproc = iproc

        from generative_inpainting_pytorch.model.networks import Generator
        self.generator = Generator(
            config={
                "input_dim": 5,
                "ngf": 32,
            },
            use_cuda=True,
            device_ids=[0],
        )
        self.generator.load_state_dict(torch.load(
            "/gpfs/data/geraslab/zphang/working/190504_infill/pretrained_models/torch_model_generator.p"
        ))

    def forward(self, masked_x, mask):
        # masked_x: normalize from [0, 1]
        # mask: 1 = selected region

        # generator input: [0, 255] / 127.5 - 1
        input_x = self.iproc.denorm_tensor(masked_x) * 255/127.5 - 1
        input_x = input_x * (1 - mask)
        _, raw_result, _ = self.generator(
            x=input_x,
            mask=mask,
        )
        result = self.iproc.norm_tensor((raw_result + 1) * 127.5/255)
        return result


class DFNInfillerWrapper(nn.Module):
    def __init__(self, iproc):
        super().__init__()
        self.iproc = iproc
        from casme.ext.dfnet import DFNet
        self.model = DFNet()
        self.model.load_state_dict(torch.load(
            "/gpfs/data/geraslab/zphang/code/DFNet/model/model_places2.pth",
            map_location=torch.device("cpu"),
        ))
        self.model.eval()

    def forward(self, masked_x, mask):
        # masked_x: normalize from [0, 1]
        # mask: 1 = selected region
        # Resize
        masked_x = F.interpolate(masked_x, 256, mode="bilinear")
        mask = F.interpolate(mask, 256, mode="bilinear")

        # desired input: [0, 1]
        # desired mask: [0, 1], 0 = masked out
        input_x = self.iproc.denorm_tensor(masked_x)
        mask_out = 1 - mask
        imgs_miss = input_x * mask_out

        result, alpha, raw = self.model(imgs_miss, mask_out)
        result, alpha, raw = result[0], alpha[0], raw[0]
        result = imgs_miss + result * mask

        result = F.interpolate(result, 224, mode="bilinear")
        result = self.iproc.norm_tensor(result)
        return result


class DummyInfiller(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, masked_x, mask):
        return masked_x * 0


class ImageProc(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.mean_tensor = nn.Parameter(
            torch.FloatTensor(self.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False,
        )
        self.std_tensor = nn.Parameter(
            torch.FloatTensor(self.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            requires_grad=False,
        )

    def forward(self):
        pass

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def norm_tensor(self, x):
        return (x - self.mean_tensor) / self.std_tensor

    def denorm_tensor(self, x):
        return x * self.std_tensor + self.mean_tensor


def get_infiller(infiller_model):
    if infiller_model == "cnn":
        return InfillerCNN(
            4, 3, [32, 64, 128, 256],
        )
    elif infiller_model == "ca_infiller":
        return CAInfillerWrapper(ImageProc(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
        ))
    elif infiller_model == "dfn_infiller":
        return DFNInfillerWrapper(ImageProc(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
        ))
    elif infiller_model == "dummy":
        return DummyInfiller()
    else:
        raise KeyError(infiller_model)


def should_train_infiller(infiller_model):
    if infiller_model == "cnn":
        return True
    elif infiller_model == "ca_infiller":
        return False
    elif infiller_model == "dfn_infiller":
        return False
    elif infiller_model == "dummy":
        return False
    else:
        raise KeyError(infiller_model)


def masker(**kwargs):
    return Masker([64, 256, 512, 1024, 2048], 64, **kwargs)
