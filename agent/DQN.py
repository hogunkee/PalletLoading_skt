import torch
from utils import *

from models import BinNet as FCQNet


class DQN_Agent():
    def __init__(self, train, model_path='', use_coordnconv=False, config=None):
        self.FCQ = FCQNet(2, config.max_levels, config.resolution**2,
                          use_coordnconv=use_coordnconv).cuda()

        if train:
            if config.do_double:
                self.FCQ_target = FCQNet(2, config.max_levels, config.resolution**2,
                                         use_coordnconv=use_coordnconv).cuda()
                self.FCQ_target.load_state_dict(self.FCQ.state_dict())
            else:
                self.FCQ_target = None

            self.FCQ.train()
            self.optimizer = torch.optim.Adam(self.FCQ.parameters(), lr=config.learning_rate)

        else:
            self.FCQ.load_state_dict(torch.load(model_path))
            self.FCQ.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.FCQ.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("# of params: %d"%params)
        
        self.config = config

    def train_on_off(self, train):
        if train: self.FCQ.train()
        else: self.FCQ.eval()

    def get_action(self, state, block, qmask,
                   with_q=False, deterministic=True, p_project=0.0):
        state_tensor = torch.FloatTensor([state]).cuda()
        block_tensor = torch.FloatTensor([block]).cuda()
        qmask_tensor = torch.FloatTensor([qmask]).cuda()

        q_value = self.FCQ(state_tensor, block_tensor, qmask_tensor)
        q_value = q_value[0].detach().cpu().numpy()
        #q_value = np.random.rand(2, q_value.shape[1], q_value.shape[2])

        use_mask = True if qmask is not None else False
        use_projection = True if p_project > 0.0 else False

        if deterministic:
            if use_mask: q_value *= qmask

            n_th, n_y, n_x = q_value.shape
            aidx = np.argmax(q_value)
            aidx_th = aidx // (n_y*n_x)
            aidx_y = (aidx % (n_y*n_x)) // n_x
            aidx_x = (aidx % (n_y*n_x)) % n_x
                
        else:
            n_th, n_y, n_x = q_value.shape

            if use_mask:
                q_masked = q_value*qmask
            else:
                q_masked = q_value

            soft_tmp = 1e-1 # 3e-1 # 1e-1
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
            return action, None
        
    def calculate_loss(self, minibatch, FCQ, FCQ_target, gamma=0.95, max_error=5.0):
        state, block, qmask, next_state, next_block, next_qmask, actions, rewards, not_done = minibatch
        state = state.type(torch.float32)
        qmask = qmask.type(torch.float32)
        next_state = next_state.type(torch.float32)
        next_qmask = next_qmask.type(torch.float32)
        actions = actions.type(torch.long)

        def get_a_prime():
            next_q = FCQ(next_state, next_block, next_qmask)
            next_q *= next_qmask

            aidx_y = next_q.max(1)[0].max(2)[0].max(1)[1]
            aidx_x = next_q.max(1)[0].max(1)[0].max(1)[1]
            aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
            return aidx_th, aidx_y, aidx_x

        B, _, n_y, n_x = state.shape

        q_values = FCQ(state, block, qmask)
        q_values = q_values * qmask
        q_values = q_values.view(-1, 2*n_y*n_x)

        indices = (actions[:, 0] * n_x*n_y + actions[:,1]*n_x + actions[:,2]).unsqueeze(-1)
        q_values = q_values.gather(1, indices)
        pred = q_values.view(-1, 1)

        a_prime = get_a_prime()
        if FCQ_target is None:
            next_q_target = FCQ(next_state, next_block, next_qmask)
        else:
            next_q_target = FCQ_target(next_state, next_block, next_qmask)
        next_q_target *= next_qmask
        q_target_s_a_prime = next_q_target[torch.arange(next_q_target.shape[0]), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        y_target = rewards + gamma * not_done * q_target_s_a_prime
        y_target = y_target.view(-1, 1).detach()
        if max_error > 0.0:
            y_target = torch.clip(y_target, pred-max_error, pred+max_error)

        #loss = criterion(y_target.detach(), pred)
        #loss = torch.clip(pred-y_target, -5.0, 5.0).pow(2).mean()
        loss = torch.mean((pred - y_target).pow(2))
        error = torch.abs(pred - y_target)
        return loss, error


    def update_network(self, minibatch):
        loss, _ = self.calculate_loss(minibatch, self.FCQ, self.FCQ_target)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.FCQ.parameters(), self.config.grad_clip)
        self.optimizer.step()

        if self.FCQ_target is not None:
            for target_param, local_param in zip(self.FCQ_target.parameters(), self.FCQ.parameters()):
                target_param.data.copy_(self.config.tau*local_param.data + (1.0-self.config.tau)*target_param.data)
        return loss.data.detach().cpu().numpy()
    
    def save_network(self, save_name, save_type="best"):
        if save_type == "best":
            torch.save(self.FCQ.state_dict(), 'results/models/%s_best.pth' % save_name)
        elif save_type == "last":
            torch.save(self.FCQ.state_dict(), 'results/models/%s_last.pth' % save_name)
        else:
            torch.save(self.FCQ.state_dict(), 'results/models/%s.pth' % save_name)

    

