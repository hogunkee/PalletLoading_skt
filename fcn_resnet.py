import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != channel:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channel//4, channel//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channel, self.expansion*channel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * channel)
                        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCQResNetSmall(nn.Module):
    def __init__(self, n_actions, in_ch, n_hidden=8, block=BasicBlock):
        super(FCQResNetSmall, self).__init__()
        self.in_channel = 8
        self.n_actions = n_actions
        num_blocks = [2, 2, 1]
        self.pad = 2

        self.conv1 = nn.Conv2d(in_ch, n_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_hidden)
        self.layer1 = self._make_layer(block, n_hidden, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*n_hidden, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*n_hidden, num_blocks[2], stride=2)

        # FC layers
        self.fully_conv = nn.Sequential(
                nn.Conv2d(4*n_hidden + 2, 8*n_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(8*n_hidden, 8*n_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(8*n_hidden, 1, kernel_size=1),
                )

        # self.upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upscore = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            #nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def _make_layer(self, block, channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channel, stride))
            self.in_channel = channel * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, block, debug=False):
        if debug:
            frames = []
            from matplotlib import pyplot as plt

        pad = self.pad
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=1)
        output_prob = []
        for r_idx in range(self.n_actions):
            if r_idx==0:
                x_rotate = x_pad
            elif r_dix==1:
                x_rotate = torch.rot90(x_pad, k=3, dims=[2, 3])

            h = F.relu(self.bn1(self.conv1(x_rotate)))
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            B, C, H, W = h.size()
            h_block = block.view(B, 2, 1, 1).repeat([1, 1, H, W])
            h_cat = torch.cat([h, h_block], axis=1)
            h = self.fully_conv(h_cat)
            h = self.upscore(h)

            if r_idx==0:
                h_after = h
            else:
                h_after = torch.rot90(h, k=1, dims=[2,3])
            h_after = h_after[:, :, pad:-pad, pad:-pad]
            output_prob.append(h_after)

            if debug:
                f = x_pad.detach().cpu().numpy()[0].transpose([1, 2, 0])
                f_rotate = x_rotate.detach().cpu().numpy()[0].transpose([1, 2, 0])

                x_re_rotate = F.grid_sample(x_rotate, flow_grid_after, align_corners=False, mode='nearest')
                f_re_rotate = x_re_rotate.detach().cpu().numpy()[0].transpose([1, 2, 0])
                frames.append([f_rotate, f_re_rotate])

        if debug:
            fig = plt.figure()
            for i in range(len(frames)):
                ax = fig.add_subplot(4, self.n_actions, i + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("%d\xb0" %(i*45))
                plt.imshow(frames[i][0][..., :3])
                ax = fig.add_subplot(4, self.n_actions, i + len(frames) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(frames[i][0][..., 3:])

                ax = fig.add_subplot(4, self.n_actions, i + 2*len(frames) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(frames[i][1][..., :3])
                ax = fig.add_subplot(4, self.n_actions, i + 3*len(frames) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(frames[i][1][..., 3:])
            plt.show()
        return torch.cat(output_prob, 1)


class FCQResNet(nn.Module):
    def __init__(self, n_actions, in_ch, n_hidden=8, block=BasicBlock):
        super(FCQResNet, self).__init__()
        self.in_channel = 8
        self.n_actions = n_actions
        num_blocks = [2, 2, 1, 1]
        self.pad = 2

        self.conv1 = nn.Conv2d(in_ch, n_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_hidden)
        self.layer1 = self._make_layer(block, n_hidden, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*n_hidden, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*n_hidden, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*n_hidden, num_blocks[3], stride=2)

        # FC layers
        self.fully_conv = nn.Sequential(
                nn.Conv2d(8*n_hidden + 2, 16*n_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(16*n_hidden, 16*n_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(16*n_hidden, 1, kernel_size=1),
                )

        # self.upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upscore = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def _make_layer(self, block, channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channel, stride))
            self.in_channel = channel * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, block, debug=False):
        if debug:
            frames = []
            from matplotlib import pyplot as plt

        B0 = x.size()[0]
        pad = self.pad
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=1)

        x_cat = x_pad.repeat([self.n_actions, 1, 1, 1])
        h = F.relu(self.bn1(self.conv1(x_cat)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        _, C, H, W = h.size()
        if self.n_actions==1:
            h_block = block.view(B0, 2, 1, 1).repeat([self.n_actions, 1, H, W])
        else:
            h_block_origin = block.view(B0, 2, 1, 1).repeat([1, 1, H, W])
            block_flipped = block[..., [1, 0]]
            h_block_flipped = block_flipped.view(B0, 2, 1, 1).repeat([1, 1, H, W])
            h_block = torch.cat([h_block_origin, h_block_flipped], axis=0)
        h_cat = torch.cat([h, h_block], axis=1)
        h = self.fully_conv(h_cat)
        h_after = self.upscore(h)
        h_after = h_after[:, :, 
                int((h_after.size()[2]-x.size()[2])/2):int((h_after.size()[2]+x.size()[2])/2),
                int((h_after.size()[3]-x.size()[3])/2):int((h_after.size()[3]+x.size()[3])/2)
                ]
        
        _, C2, H2, W2 = h_after.size() 
        output_prob = h_after.view(self.n_actions, -1, H2, W2).permute([1, 0, 2, 3])
        return output_prob


    def forward2(self, x, block, debug=False):
        if debug:
            frames = []
            from matplotlib import pyplot as plt

        B0 = x.size()[0]
        pad = self.pad
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=1)

        x_rotate_list = []
        for r_idx in range(self.n_actions):
            x_rotate = torch.rot90(x_pad, k=r_idx * 3, dims=[2, 3])
            x_rotate_list.append(x_rotate)
        x_cat = torch.cat(x_rotate_list, axis=0)

        h = F.relu(self.bn1(self.conv1(x_cat)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        _, C, H, W = h.size()
        h_block = block.view(B0, 2, 1, 1).repeat([self.n_actions, 1, H, W])
        h_cat = torch.cat([h, h_block], axis=1)
        h = self.fully_conv(h_cat)
        h = self.upscore(h)
        
        output_prob = []
        for r_idx in range(self.n_actions):
            h_after = h[r_idx * B0: (r_idx + 1) * B0]
            h_after = torch.rot90(h_after, k=r_idx * 1, dims=[2,3])
            h_after = h_after[:, :, pad:-pad, pad:-pad]
            output_prob.append(h_after)

        return torch.cat(output_prob, 1)

