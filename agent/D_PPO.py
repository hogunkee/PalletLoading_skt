import time
import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Categorical

from models import DiscreteActor
from models import BinNet as Critic
from itertools import chain
from utils import *

class D_PPO_Agent:
    def __init__(self, max_levels, resolution, train,
                 learning_rate=3e-4,
                 model_path='',
                 use_coordnconv=False,
                 gamma=0.99,
                 lam=0.95,
                 epsilon=0.2,
                 device='cuda',
                 num_updates=1,
                 ):

        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.lr = learning_rate
        self.num_updates = num_updates

        self.pi = DiscreteActor(2, max_levels, resolution**2,
                                use_coordnconv=use_coordnconv).cuda()
        self.V = Critic(2, max_levels, resolution**2,
                        use_coordnconv=use_coordnconv).cuda()
        if train:
            params = chain(self.pi.parameters(), self.V.parameters())
            self.optimizer = Adam(params, lr=self.lr)
        
        else:
            critic_path = model_path[0]
            actor_path = model_path[1]
            self.V.load_state_dict(torch.load(critic_path))
            self.pi.load_state_dict(torch.load(actor_path))
            self.V.eval()
            self.pi.eval()

        self.device = device
        actor_parameters = filter(lambda p: p.requires_grad, self.pi.parameters())
        critic_parameters = filter(lambda p: p.requires_grad, self.V.parameters())
        actor_params = sum([np.prod(p.size()) for p in actor_parameters])
        critic_params = sum([np.prod(p.size()) for p in critic_parameters])
        params = actor_params + critic_params
        print("# of params: %d"%params)

    def train_on_off(self, train):
        if train:
            self.pi.train()
            self.V.train()
        else:
            self.pi.train()
            self.V.train() 

    def get_action(self, state, block, 
                   with_q=False, deterministic=True, q_mask=None, p_project=0.0):
        state_tensor = torch.FloatTensor([state]).cuda()
        block_tensor = torch.FloatTensor([block]).cuda()
        
        _, n_y, n_x = state.shape
        action_idx, probs, _ = self.pi(state_tensor, block_tensor, deterministic=deterministic, q_mask=q_mask)
        action_th = action_idx // (n_y*n_x)
        action_y = (action_idx % (n_y*n_x)) // n_x
        action_x = (action_idx % (n_y*n_x)) % n_x

        action = [action_th.view(-1).item(), action_y.view(-1).item(), action_x.view(-1).item()]

        if np.random.random() < p_project:
            action = action_projection(state, block, action)
        
        val = self.V(state_tensor, block_tensor)
        val = val.cpu().detach().numpy()

        probs = probs.gather(1, action_idx)
        probs = probs.cpu().detach().numpy()
        
        return action, probs, val

    def learn(self, batch):
        # unroll batch
        state, block, actions, target_v, A, old_probs, _ = batch  
        B, _, n_y, n_x = state.shape
        for i in range(self.num_updates):
            action, probs, log_probs = self.pi(state, block)
            indices = (actions[:, 0] * n_x*n_y + actions[:,1]*n_x + actions[:,2]).unsqueeze(-1)

            entropy = Categorical(probs).entropy().sum(0, keepdim=True)
            probs = probs.gather(1, action.long())

            # compute prob ratio
            r = torch.exp(torch.log(probs) - torch.log(old_probs))

            # clipped loss
            clipped_r = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon)

            # surrogate objective
            A = A.view(-1, 2*n_y*n_x).gather(1, indices.long())
            single_step_obj = torch.min(r * A, clipped_r * A)
            pi_loss = -torch.mean(single_step_obj) - 0.1 * entropy

            v = self.V(state, block).view(-1, 2*n_y*n_x)
            v = v.gather(1, indices.long())
            
            target_v = (probs * target_v.view(-1, 2*n_y*n_x)).sum(dim=1, keepdims=True)
            
            V_loss = torch.mean((v - target_v) ** 2)

            loss = pi_loss + V_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
