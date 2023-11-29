import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class BinNet(nn.Module):
    def __init__(self, n_rotations, in_channels, out_dim, n_hidden=[16,32,64], use_coordnconv=False):
        super(BinNet, self).__init__()
        self.n_rotations = n_rotations

        if use_coordnconv:
            self.add_coords = AddCoords()
            in_channels = in_channels + 2 + 2 + 2
        else:
            self.add_coords = None
            in_channels = in_channels + 2 + 2


        self.conv1 = nn.Conv2d(in_channels, n_hidden[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(n_hidden[0])
        self.conv2 = nn.Conv2d(n_hidden[0], n_hidden[1], kernel_size=2, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(n_hidden[1])
        self.conv3 = nn.Conv2d(n_hidden[1], n_hidden[2], kernel_size=2, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(n_hidden[2])

        self.conv_final = nn.Conv2d(n_hidden[2], 1, kernel_size=1)

        self.upscore = nn.Sequential(
            nn.Linear(4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim*n_rotations),
            nn.ReLU(),
            #nn.Sigmoid(),
        )

    def forward(self, x, block, qmask):
        B0, _, H, W = x.size()
        if self.add_coords is not None:
            x = self.add_coords(x)

        x_block = block[..., None, None]
        x_block = (torch.ceil(x_block*H)/H).repeat(1,1,H,W)
        x_cat = torch.cat([x, x_block, qmask], axis=1)

        h = F.relu(self.bn1(self.conv1(x_cat)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        h = self.conv_final(h)
        h = torch.flatten(h, start_dim=1)
        output_prob = self.upscore(h)
        output_prob = output_prob.view(-1, self.n_rotations, H, W)

        # output_scale = 1e2
        # output_prob = torch.sigmoid(output_prob)*output_scale
        return output_prob



class DiscreteActor(nn.Module):
    def __init__(self, n_rotations, in_channels, out_dim, n_hidden=[16,32,64], use_coordnconv=False):
        super(DiscreteActor, self).__init__()
        self.n_rotations = n_rotations

        if use_coordnconv:
            self.add_coords = AddCoords()
            in_channels = in_channels + 2 + 2 + 2
        else:
            self.add_coords = None
            in_channels = in_channels + 2 + 2


        self.conv1 = nn.Conv2d(in_channels, n_hidden[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(n_hidden[0])
        self.conv2 = nn.Conv2d(n_hidden[0], n_hidden[1], kernel_size=2, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(n_hidden[1])
        self.conv3 = nn.Conv2d(n_hidden[1], n_hidden[2], kernel_size=2, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(n_hidden[2])

        self.conv_final = nn.Conv2d(n_hidden[2], 1, kernel_size=1)

        self.upscore = nn.Sequential(
            nn.Linear(4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim*n_rotations),
            nn.ReLU(),
            #nn.Sigmoid(),
        )

    def forward(self, x, block, qmask, deterministic=False, tsallis=False, q_prime=1.2):
        B0, _, H, W = x.size()
        if self.add_coords is not None:
            x = self.add_coords(x)

        x_block = block[..., None, None]
        x_block = (torch.ceil(x_block*H)/H).repeat(1,1,H,W)
        x_cat = torch.cat([x, x_block, qmask], axis=1)

        h = F.relu(self.bn1(self.conv1(x_cat)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        h = self.conv_final(h)
        h = torch.flatten(h, start_dim=1)
        action_logits = self.upscore(h)
     
        q_mask = qmask.view(action_logits.shape).to(x.device)
        #min_logits, _ = torch.min(action_logits, dim=1, keepdim=True)
        min_logits = torch.zeros_like(action_logits)            
        action_logits = torch.where(q_mask > 0, action_logits, min_logits)

        soft_tmp = 1e-1
        action_probs = F.softmax(action_logits/soft_tmp, dim=1)
        action_dist = Categorical(action_probs)

        if deterministic:
            actions = torch.argmax(action_probs, 1).view(-1, 1)
            log_action_probs = None
        else:        
            actions = action_dist.sample().view(-1, 1)
            # Avoid numerical instability.
            z = (action_probs == 0.0).float() * 1e-8
            log_action_probs = torch.log(action_probs + z)
            if tsallis and q_prime != 1.0:
                log_action_probs = torch_log_q(torch.exp(log_action_probs), 2.0-q_prime)
        return actions, action_probs, log_action_probs

def torch_log_q(x, q):
    safe_x = torch.max(x, torch.tensor(1e-6))
    log_q_x = (torch.pow(safe_x, 1 - q) - 1) / (1 - q)
    return log_q_x

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

