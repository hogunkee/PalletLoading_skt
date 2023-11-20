import cv2
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn


#criterion = nn.SmoothL1Loss(reduction='mean').cuda()
criterion = nn.MSELoss(reduction='mean').cuda()

def sample_trajectories(blocks, actions, current_state, resolution):
    trajectories = []
    for block, action in zip(blocks, actions):
        # make box region #
        cy, cx, _ = action
        by, bx = np.round(np.array(block) * resolution).astype(int)
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
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1], actions[:, 2]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.95):
    state = minibatch[0]
    block = minibatch[1]
    next_state = minibatch[2]
    next_block = minibatch[3]
    actions = minibatch[4].type(torch.long)
    rewards = minibatch[5]
    not_done = minibatch[6]
    batch_size = state.size()[0]

    def get_a_prime():
        next_q = FCQ(next_state, next_block)
        #next_q = FCQ_target(next_state, next_block)
        aidx_y = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_x = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_y, aidx_x

    a_prime = get_a_prime()

    next_q_target = FCQ_target(next_state, next_block)
    q_target_s_a_prime = next_q_target[torch.arange(next_q_target.shape[0]), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state, block)
    pred = q_values[torch.arange(q_values.shape[0]), actions[:, 0], actions[:, 1], actions[:, 2]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


def get_block_bound(center_y, center_x, block_y, block_x):    
    block_y, block_x = math.ceil(block_y), math.ceil(block_x)
    min_y, max_y = math.floor(center_y-block_y/2), math.floor(center_y+block_y/2)
    min_x, max_x = math.floor(center_x-block_x/2), math.floor(center_x+block_x/2)
    return min_y, max_y, min_x, max_x

def generate_cumulative_state(state):
    max_level = np.shape(state)[0]
    cum_state = np.zeros_like(state)
    for i in reversed(range(max_level)):
        if i == max_level-1:
            cum_state[i] = np.copy(state[i])
        else:
            cum_state[i] = np.clip(np.copy(state[i])+np.copy(cum_state[i+1]), 0.0, 1.0)
    return cum_state


def generate_floor_mask(state, block, pre_mask=None):
    resolution = np.shape(state)[1]
    if pre_mask is None:
        mask = np.ones((2,resolution,resolution))
    else:
        mask = np.copy(pre_mask)

    by, bx = block
    by, bx = math.ceil(by*resolution), math.ceil(bx*resolution)

    max_level = np.shape(state)[0]
    cum_state = generate_cumulative_state(state)
    level_map = np.sum(cum_state, axis=0)

    min_packed_ratio, empty_level = 0.75, -1
    for i in range(0,max_level-1):
        if np.mean(cum_state[i]) < min_packed_ratio:
            empty_level = i+1
            break

    if empty_level > 0:   
        for y_ in range(resolution):
            for x_ in range(resolution):
                if mask[0,y_,x_] == 0: continue
                min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
                min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

                box_placed = np.zeros(np.shape(state[0]))
                box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1

                curr_map = level_map + box_placed
                if len(np.where(np.array(curr_map)>empty_level)[0]) > 0:
                    mask[0,y_,x_] = 0

        for y_ in range(resolution):
            for x_ in range(resolution):
                if mask[1,x_,y_] == 0: continue
                min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
                min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

                box_placed = np.zeros(np.shape(state[0]))
                box_placed[max(min_x,0): max_x, max(min_y,0): max_y] = 1

                curr_map = level_map + box_placed
                if len(np.where(np.array(curr_map)>empty_level)[0]) > 0:
                    mask[1,x_,y_] = 0

    if np.sum(mask) == 0:
        return pre_mask
    else:
        return mask


def generate_bound_mask(state, block):
    resolution = np.shape(state)[1]
    mask = np.ones((2,resolution,resolution))

    by, bx = block
    by, bx = math.ceil(by*resolution), math.ceil(bx*resolution)

    max_level = np.shape(state)[0]
    cum_state = generate_cumulative_state(state)
    level_map = np.sum(cum_state, axis=0)

    for y_ in range(resolution):
        min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
        if min_y < 0 or max_y > resolution:
            mask[0,y_,:] = 0
            mask[1,:,y_] = 0

    for x_ in range(resolution):
        min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)
        if min_x < 0 or max_x > resolution:
            mask[0,:,x_] = 0
            mask[1,x_,:] = 0

    for y_ in range(resolution):
        for x_ in range(resolution):
            if mask[0,y_,x_] == 0: continue
            min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
            min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

            box_placed = np.zeros(np.shape(state[0]))
            box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1

            curr_map = level_map + box_placed
            if len(np.where(np.array(curr_map)>max_level)[0]) > 0:
                mask[0,y_,x_] = 0

    for y_ in range(resolution):
        for x_ in range(resolution):
            if mask[1,x_,y_] == 0: continue
            min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
            min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

            box_placed = np.zeros(np.shape(state[0]))
            box_placed[max(min_x,0): max_x, max(min_y,0): max_y] = 1

            curr_map = level_map + box_placed
            if len(np.where(np.array(curr_map)>max_level)[0]) > 0:
                mask[1,x_,y_] = 0
    return mask
    

def action_projection(state, block, action, box_norm=True):
    action_rot = action[0]
    action_pos = np.array(action[1:])
    cy, cx = action_pos
    resolution = np.shape(state)[1]

    if action_rot == 0:
        by, bx = block
    elif action_rot == 1:
        bx, by = block

    if box_norm:
        by *= resolution
        bx *= resolution
        
    next_block_bound = get_block_bound(cy, cx, by, bx)
    min_y, max_y, min_x, max_x = next_block_bound

    max_level = np.shape(state)[0]
    cum_state = generate_cumulative_state(state)
    level_map = np.sum(cum_state, axis=0)

    box_level0 = np.max(level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1
    if box_level0 > max_level:
        return [action_rot, cy, cx]
    

    while True:
        proj_y = project_axis("y",
                              min_y, max_y, min_x, max_x,
                              level_map, box_level0)

        proj_x = project_axis("x",
                              min_y-proj_y, max_y-proj_y, min_x, max_x,
                              level_map, box_level0)

        min_y, max_y = min_y-proj_y, max_y-proj_y
        min_x, max_x = min_x-proj_x, max_x-proj_x

        cy -= proj_y
        cx -= proj_x
        if proj_y == 0 and proj_x == 0: break

    return [action_rot, cy, cx]

def project_axis(axis, min_y, max_y, min_x, max_x, level_map, box_level0):
    proj_ = 0        
    while True:
        if axis == "y":
            min_y_, max_y_ = min_y-proj_, max_y-proj_
            min_x_, max_x_ = min_x, max_x
        elif axis == "x":
            min_y_, max_y_ = min_y, max_y
            min_x_, max_x_ = min_x-proj_, max_x-proj_
        if min(min_y_, min_x_) < 0: break

        box_level = np.max(level_map[max(min_y_,0):max_y_,max(min_x_,0):max_x_]) + 1
        if box_level != box_level0: break

        proj_ += 1
    if proj_ > 0: proj_ -= 1
    return proj_