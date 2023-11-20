import torch
from utils import *

from models import BinNet as FCQNet


class DQN_Agent():
    def __init__(self, max_levels, resolution, train,
                 learning_rate=3e-4, model_path='',
                 continue_learning=False, do_double=True):
        self.FCQ = FCQNet(2, max_levels, resolution**2).cuda()

        if (not train) or continue_learning:
            self.FCQ.load_state_dict(torch.load(model_path))

        if train and do_double:
            self.FCQ_target = FCQNet(2, max_levels, resolution**2).cuda()
            self.FCQ_target.load_state_dict(self.FCQ.state_dict())

            self.calculate_loss = calculate_loss_double_fcdqn
            self.FCQ.train()

            self.optimizer = torch.optim.Adam(self.FCQ.parameters(), lr=learning_rate)

        else:
            self.FCQ.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.FCQ.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("# of params: %d"%params)

    def train_on_off(self, train):
        if train:
            self.FCQ.train()
        else:
            self.FCQ.eval()

    def get_action(self, state, block,
                   with_q=False, deterministic=True, q_mask=None, p_project=0.0):
        state_tensor = torch.FloatTensor([state]).cuda()
        block_tensor = torch.FloatTensor([block]).cuda()

        q_value = self.FCQ(state_tensor, block_tensor)
        q_value = q_value[0].detach().cpu().numpy()

        use_mask = True if q_mask is not None else False
        use_projection = True if p_project > 0.0 else False

        if deterministic:
            if use_mask: q_value *= q_mask

            aidx_y = q_value.max(0).max(1).argmax()
            aidx_x = q_value.max(0).max(0).argmax()
            aidx_th = q_value.argmax(0)[aidx_y, aidx_x]
                
        else:
            n_th, n_y, n_x = q_value.shape

            if use_mask:
                q_masked = q_value*q_mask

            soft_tmp = 1e0 # 3e-1 # 1e-1
            q_probs = q_masked.reshape((-1,))
            q_probs = np.exp((q_probs-q_probs.max())/soft_tmp)
            q_probs = q_probs / q_probs.sum()

            aidx = np.random.choice(len(q_probs), 1, p=q_probs)[0]
            aidx_th = aidx // (n_y*n_x)
            aidx_y = (aidx % (n_y*n_x)) // n_x
            aidx_x = (aidx % (n_y*n_x)) % n_x

        action = [aidx_th, aidx_y, aidx_x]

        if np.random.random() < p_project:
            action = action_projection(state, block, action)

        if with_q:
            return action, q_value
        else:
            return action
        

    def update_network(self, minibatch, tau=1e-3, clip=1.0):
        loss, _ = self.calculate_loss(minibatch, self.FCQ, self.FCQ_target)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.FCQ.parameters(), clip)
        self.optimizer.step()

        for target_param, local_param in zip(self.FCQ_target.parameters(), self.FCQ.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        return loss.data.detach().cpu().numpy()
    

