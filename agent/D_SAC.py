import torch
import copy
import time
from torch.optim import Adam

from models import DiscreteActor
from models import BinNet as Critic
from utils import *


class D_SAC_Agent:
    def __init__(self, max_levels, resolution, train,
                 learning_rate=3e-4,
                 model_path=[],
                 use_coordnconv=False,
                 gamma=0.99,
                 target_entropy_ratio=0.98,
                 device='cuda',
                 ):
        self.gamma = gamma

        # networks definition
        # pi : actor network, Q : 2 critic network
        self.pi = DiscreteActor(2, max_levels, resolution**2,
                                use_coordnconv=use_coordnconv).cuda()
        self.Q = Critic(2, max_levels, resolution**2,
                        use_coordnconv=use_coordnconv).cuda()
        
        if train:
            self.pi.train()
            self.Q.train()

            # target networks
            self.target_Q = copy.deepcopy(self.Q).cuda()

            freeze(self.target_Q)

            self.critic_optimizer = Adam(self.Q.parameters(), lr=learning_rate)
            self.actor_optimizer = Adam(self.pi.parameters(), lr=learning_rate)

            self.target_entropy = \
                -np.log(1.0 / resolution*resolution*2) * target_entropy_ratio

            # self.log_alpha = torch.zeros(1, requires_grad=True).to(device)
            # self.alpha = self.log_alpha.exp()
            # self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
            self.alpha = 0.2
            self.device = device

        else:
            critic_path = model_path[0]
            actor_path = model_path[1]
            self.Q.load_state_dict(torch.load(critic_path))
            self.pi.load_state_dict(torch.load(actor_path))
            self.Q.eval()
            self.pi.eval()

        actor_parameters = filter(lambda p: p.requires_grad, self.pi.parameters())
        critic_parameters = filter(lambda p: p.requires_grad, self.Q.parameters())
        actor_params = sum([np.prod(p.size()) for p in actor_parameters])
        critic_params = sum([np.prod(p.size()) for p in critic_parameters])
        params = actor_params + critic_params
        print("# of params: %d"%params)
    
    def train_on_off(self, train):
        if train:
            self.pi.train()
            self.Q.train()
        else:
            self.pi.train()
            self.Q.train() 

    def get_action(self, state, block, 
                   with_q=False, deterministic=True, q_mask=None, p_project=0.0):
        state_tensor = torch.FloatTensor([state]).cuda()
        block_tensor = torch.FloatTensor([block]).cuda()
        
        _, n_y, n_x = state.shape
        action_idx,_,_ = self.pi(state_tensor, block_tensor, deterministic=deterministic, q_mask=q_mask)
        action_th = action_idx // (n_y*n_x)
        action_y = (action_idx % (n_y*n_x)) // n_x
        action_x = (action_idx % (n_y*n_x)) % n_x

        action = [action_th.view(-1).item(), action_y.view(-1).item(), action_x.view(-1).item()]

        if np.random.random() < p_project:
            action = action_projection(state, block, action)
        
        if with_q:
            q_value = self.Q(state_tensor, block_tensor)
            q_value = q_value[0].detach().cpu().numpy()
            if q_mask is not None:
                q_value *= q_mask
            return action, q_value
        else:
            return action
  
    def calculate_critic_loss(self, batch, weights):
        state, block, next_state, next_block, next_qmask, actions, rewards, not_done = batch
        state = state.type(torch.float32)
        next_state = next_state.type(torch.float32)
        next_qmask = next_qmask.type(torch.float32)
        actions = actions.type(torch.long)
        
        B, _, n_y, n_x = state.shape
        indices = (actions[:, 0] * n_x*n_y + actions[:,1]*n_x + actions[:,2]).unsqueeze(-1)

        current_q = self.Q(state, block).view(-1, 2*n_y*n_x)
        current_q = current_q.gather(1, indices)

        with torch.no_grad():
            _, action_probs, log_action_probs = self.pi(next_state, next_block, q_mask=next_qmask)
            next_q = self.target_Q(next_state, next_block).view(B,-1)
            next_q = (action_probs * (
                next_q - self.alpha * log_action_probs
            )).sum(dim=1, keepdim=True)
        target_q = rewards + not_done * self.gamma * next_q

        # Critic loss is mean squared TD errors with priority weights.
        q_loss = torch.mean((current_q - target_q).pow(2) * weights)

        return q_loss

    def calculate_policy_loss(self, batch, weights):
        state, block, next_state, next_block, next_qmask, actions, rewards, not_done = batch
        B, _, n_y, n_x = state.shape
        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.pi(state, block)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q = self.Q(state, block).view(B, -1)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(q * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()
    
    def calculate_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def learn(self, batch, tau=1e-3, clip=1.0):
        q_loss = self.calculate_critic_loss(batch, 1.)
        policy_loss, entropies = self.calculate_policy_loss(batch, 1.)
        # entropy_loss = self.calculate_entropy_loss(entropies, 1.)
        
        update_params(self.critic_optimizer, q_loss)
        self.soft_update(self.Q, self.target_Q, tau=tau)
        update_params(self.actor_optimizer, policy_loss)
        # update_params(self.alpha_optim, entropy_loss)
        # self.alpha = self.log_alpha.exp()
        loss = (q_loss + policy_loss).cpu().detach().numpy()

        return loss

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()