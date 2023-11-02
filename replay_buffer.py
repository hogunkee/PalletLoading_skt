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
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, block, action, next_state, next_block, reward, done):
        self.state[self.ptr] = np.array(state, dtype=np.uint8)
        self.block[self.ptr] = block 
        self.next_state[self.ptr] = np.array(next_state, dtype=np.uint8)
        self.next_block[self.ptr] = next_block 
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
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        ]

        return data_batch


