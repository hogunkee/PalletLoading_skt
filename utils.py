import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn

criterion = nn.SmoothL1Loss(reduction='mean').cuda()

def sample_trajectories(blocks, actions, current_state, resolution):
    trajectories = []
    for block, action in zip(blocks, actions):
        # make box region #
        cy, cx = action
        #by, bx = self.next_block
        by, bx = np.round(np.array(self.block) * resolution).astype(int)
        min_y = np.round(cy - (by-1e-5)/2).astype(int)
        min_x = np.round(cx - (bx-1e-5)/2).astype(int)
        max_y = np.round(cy + (by-1e-5)/2).astype(int)
        max_x = np.round(cx + (bx-1e-5)/2).astype(int)

        state_copy = current_state.copy()
        state_copy[0, min_y: max_y, min_x: max_x] = 0
        traj = (state_copy, block, action)
        trajectories.append(traj)
    return trajectories

def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def smoothing_log_same(log_data, log_freq):
    return np.concatenate([np.array([np.nan] * (log_freq-1)), np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq])

def combine_batch(minibatch, data):
    try:
        combined = []
        if minibatch is None:
            for i in range(len(data)):
                combined.append(data[i].unsqueeze(0))
        else:
            for i in range(len(minibatch)):
                combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    except:
        print(i)
        print('minibatch:', len(minibatch))
        print(minibatch[i].shape)
        print('data:', len(data))
        print(data[i].shape)
        print(minibatch[i])
        print(data[i])
    return combined


## FCDQN Loss ##
def calculate_loss_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.5):
    state = minibatch[0]
    block = minibatch[1]
    next_state = minibatch[2]
    next_block = minibatch[3]
    actions = minibatch[4].type(torch.long)
    rewards = minibatch[5]
    not_done = minibatch[6]
    batch_size = state.size()[0]

    next_q = FCQ_target(next_state, next_block)
    next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
    #next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = FCQ(state, block)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.5):
    state = minibatch[0]
    block = minibatch[1]
    next_state = minibatch[2]
    next_block = minibatch[3]
    actions = minibatch[4].type(torch.long)
    rewards = minibatch[5]
    not_done = minibatch[6]
    batch_size = state.size()[0]

    def get_a_prime_pixel():
        next_q = FCQ(next_state, next_block)
        next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    def get_a_prime():
        next_q = FCQ(next_state, next_block)
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    a_prime = get_a_prime()

    next_q_target = FCQ_target(next_state, next_block)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
    #next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    #q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state, block)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

