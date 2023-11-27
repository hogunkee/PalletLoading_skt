import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer(object):
    def __init__(self, state_dim, block_dim=2, max_size=int(5e5), dim_reward=1, dim_action=2):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        self.dim_action = dim_action

        self.state = np.zeros([max_size] + list(state_dim), dtype=np.uint8)
        self.block = np.zeros((max_size, block_dim))
        self.next_state = np.zeros([max_size] + list(state_dim), dtype=np.uint8)
        self.next_block = np.zeros((max_size, block_dim))
        self.next_qmask = np.zeros([max_size,2] + list(state_dim)[1:], dtype=np.uint8)
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, block, action, next_state, next_block, next_qmask, reward, done):
        self.state[self.ptr] = np.array(state, dtype=np.uint8)
        self.block[self.ptr] = block 
        self.next_state[self.ptr] = np.array(next_state, dtype=np.uint8)
        self.next_block[self.ptr] = next_block 
        self.next_qmask[self.ptr] = np.array(next_qmask, dtype=np.uint8)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        data_batch = [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.block[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_block[ind]).to(self.device),
            torch.FloatTensor(self.next_qmask[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        ]

        return data_batch


class OnPolicyMemory:
    def __init__(self,
                 dimS,
                 dimA,
                 gamma,
                 lam,
                 lim
                 ):

        self._obs_mem = np.zeros(shape=[lim] + dimS)
        self._block_mem = np.zeros(shape=(lim, 2))
        self._act_mem = np.zeros(shape=(lim, dimA))
        self._rew_mem = np.zeros(shape=(lim,))
        self._val_mem = np.zeros(shape=[lim, 2] + dimS[1:])
        self._prob_mem = np.zeros(shape=(lim,))
        self._qmask_mem = np.zeros(shape=[lim, 2] + dimS[1:])

        # memory of cumulative rewards which are MC-estimates of the current value function
        self._target_v_mem = np.zeros(shape=[lim, 2] + dimS[1:])
        # memory of GAE($\lambda$)-estimate of the current advantage function
        self._adv_mem = np.zeros(shape=[lim, 2] + dimS[1:])

        self._gamma = gamma
        self._lam = lam

        self._lim = lim    # current size of the memory
        self._size = 0
        self._ep_start = 0
        self._head = 0       # position to save next transition sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self,):
        self._size, self._ep_start, self._head = 0, 0, 0

    def append(self, state, block, action, reward, value, prob, n_qmask):

        assert self._head < self._lim
        self._obs_mem[self._head, :] = state
        self._block_mem[self._head, :] = block
        self._act_mem[self._head, :] = action
        self._rew_mem[self._head] = reward
        self._val_mem[self._head] = value
        self._prob_mem[self._head] = prob
        self._qmask_mem[self._head] = n_qmask

        self._head += 1
        self._size += 1

        return

    def load(self):
        # load samples when the memory is full
        assert self._size == self._lim

        states = self._obs_mem[:]
        blocks = self._block_mem[:]
        actions = self._act_mem[:]
        target_v = self._target_v_mem[:]
        GAEs = self._adv_mem[:]
        probs = self._prob_mem[:]
        next_qmask = self._qmask_mem[:]

        # apply advantage normalization trick
        GAEs = (GAEs - np.mean(GAEs)) / np.std(GAEs)

        data_batch = [
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(blocks).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(target_v).to(self.device),
            torch.FloatTensor(GAEs).to(self.device),
            torch.FloatTensor(probs).to(self.device),
            torch.FloatTensor(next_qmask).to(self.device),
        ]

        return data_batch

    def compute_values(self, v_last):
        # compute advantage estimates & target values at the end of each episode
        # $v = 0$ if $s_T$ is terminal, else $v = \hat{V}(s_T)$
        gamma = self._gamma
        lam = self._lam
        start = self._ep_start   # 1st step of the epi, corresponds to t = 0
        idx = self._head - 1      # last step of the epi, corresponds to t = T - 1

        R, H, W = v_last.shape
        v_last = v_last.view(-1).cpu().detach().numpy()
        v = np.zeros([idx - start +2, R*H*W])
        v[-1] = v_last
        v[:-1] = self._val_mem[start: idx + 1].reshape(-1, R*H*W)

        # compute TD-error based on value estimate
        delta = np.tile(np.expand_dims((self._rew_mem[start: idx + 1]),1), R*H*W) + gamma * v[1:] - v[:-1]

        # backward calculation of cumulative rewards & GAE
        next_GAE = np.zeros([R, H, W])
        # for a truncated episode, the last reward is set to 0
        # otherwise, the last reward is set to $\hat{V}(s_T)$
        next_R = v_last.reshape(-1, R*H*W)

        for t in range(idx, start - 1, -1):
            self._target_v_mem[t] = (self._rew_mem[t] + gamma * next_R).reshape(-1, R, H, W)
            self._adv_mem[t] = (delta[t - start] + (gamma * lam) * next_GAE.reshape(-1, R*H*W)).reshape(-1, R, H ,W)

            next_R = self._target_v_mem[t].reshape(-1, R*H*W)
            next_GAE = self._adv_mem[t]

        self._ep_start = self._head
        return

    def sample(self, batch_size):
        """
        sample a mini-batch from the buffer
        :param batch_size: size of a mini-batch to sample
        :return: mini-batch of transition samples
        """

        rng = np.random.default_rng()
        idxs = rng.choice(self._lim, batch_size)

        states = self._obs_mem1[idxs, :]
        blocks = self._block_mem[idxs, :]
        actions = self._act_mem[idxs, :]
        target_vals = self._target_v_mem[idxs]
        GAEs = self._adv_mem[idxs]
        probs = self._prob_mem[idxs]
        next_qmask = self._qmask_mem[idxs]

        data_batch = [
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(blocks).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(target_vals).to(self.device),
            torch.FloatTensor(GAEs).to(self.device),
            torch.FloatTensor(probs).to(self.device),
            torch.FloatTensor(next_qmask).to(self.device),
        ]

        return data_batch