import math
import pickle
import numpy as np

import torch
import torch.nn as nn

#criterion = nn.SmoothL1Loss(reduction='mean').cuda()
criterion = nn.MSELoss(reduction='mean').cuda()


def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def smoothing_log_same(log_data, log_freq):
    return np.concatenate([np.array([np.nan] * (log_freq-1)), np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq])


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


def generate_floor_mask(state, block, pre_mask=None, min_packed_ratio=0.70):
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

    for level_limit in range(max_level):
        if pre_mask is None:
            mask = np.ones((2,resolution,resolution))
        else:
            mask = np.copy(pre_mask)

        if level_limit == 0:
            previous_packed_ratio = 1.0
        else:
            previous_packed_ratio = np.mean(cum_state[level_limit-1])

        if np.mean(cum_state[level_limit]) >= previous_packed_ratio*min_packed_ratio:
            continue

        for y_ in range(resolution):
            for x_ in range(resolution):
                if mask[0,y_,x_] == 0: continue
                min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
                min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

                box_placed = np.zeros(np.shape(state[0]))
                box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1

                curr_map = level_map + box_placed
                if len(np.where(np.array(curr_map)>level_limit)[0]) > 0:
                    mask[0,y_,x_] = 0

        for y_ in range(resolution):
            for x_ in range(resolution):
                if mask[1,x_,y_] == 0: continue
                min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
                min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

                box_placed = np.zeros(np.shape(state[0]))
                box_placed[max(min_x,0): max_x, max(min_y,0): max_y] = 1

                curr_map = level_map + box_placed
                if len(np.where(np.array(curr_map)>level_limit)[0]) > 0:
                    mask[1,x_,y_] = 0

        if np.sum(mask) > 0: break

    if np.sum(mask) == 0:
        return pre_mask
    else:
        return mask

def generate_solid_mask(state, block, pre_mask, min_packed_ratio=0.70):
    resolution = np.shape(state)[1]
    if pre_mask is None:
        mask = np.ones((2,resolution,resolution))
    else:
        mask = np.copy(pre_mask)

    by, bx = block
    by, bx = math.ceil(by*resolution), math.ceil(bx*resolution)

    cum_state = generate_cumulative_state(state)
    level_map = np.sum(cum_state, axis=0)

    for y_ in range(resolution):
        for x_ in range(resolution):
            if mask[0,y_,x_] == 0: continue

            min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
            min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

            box_level = np.max(level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1
            if box_level == 1: continue

            packed_ratio = np.mean(state[int(box_level)-2,max(min_y,0):max_y,max(min_x,0):max_x])
            if packed_ratio < min_packed_ratio:
                mask[0,y_,x_] = 0

    for y_ in range(resolution):
        for x_ in range(resolution):
            if mask[1,x_,y_] == 0: continue
            min_y, max_y = math.floor(y_-by/2), math.floor(y_+by/2)
            min_x, max_x = math.floor(x_-bx/2), math.floor(x_+bx/2)

            box_level = np.max(level_map[max(min_x,0):max_x,max(min_y,0):max_y]) + 1
            if box_level == 1: continue

            packed_ratio = np.mean(state[int(box_level)-2,max(min_x,0):max_x,max(min_y,0):max_y])
            if packed_ratio < min_packed_ratio:
                mask[1,x_,y_] = 0

    if np.sum(mask) == 0:
        mask = np.copy(pre_mask)

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


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def convert_full_demonstrations(demo_buffer, New_Buffer, demo_name,
                                use_floor_mask=False, max_levels=1, resolution=10):
    n_demos = demo_buffer.size
    new_buffer = New_Buffer(
        [max_levels, resolution, resolution],
        2, dim_action=3, max_size=int(n_demos)
    )

    for i in range(n_demos):
        curr_state, curr_block = demo_buffer.state[i], demo_buffer.block[i]
        curr_nextstate, curr_nextblock = demo_buffer.next_state[i], demo_buffer.next_block[i]

        if max_levels > 1:
            curr_state = np.concatenate(
                (
                    curr_state,
                    np.zeros((max_levels-1, resolution, resolution))
                ), axis=0
            )
            curr_nextstate = np.concatenate(
                (
                    curr_nextstate,
                    np.zeros((max_levels-1, resolution, resolution))
                ), axis=0
            )

        curr_qmask = generate_bound_mask(curr_state, curr_block)
        if use_floor_mask:
            curr_qmask = generate_floor_mask(curr_state, curr_block, curr_qmask)

        curr_nextqmask = generate_bound_mask(curr_nextstate, curr_nextblock)
        if use_floor_mask:
            curr_nextqmask = generate_floor_mask(curr_nextstate, curr_nextblock, curr_nextqmask)

        if max_levels == 1:
            curr_state = curr_state[0]
            curr_nextstate = curr_nextstate[0]

        curr_action, curr_reward, curr_done = demo_buffer.action[i], demo_buffer.reward[i], False
        new_buffer.add(curr_state, curr_block, curr_qmask, curr_action, \
                       curr_nextstate, curr_nextblock, curr_nextqmask, curr_reward, curr_done)
    
    with open(demo_name, "wb") as f:
        pickle.dump(new_buffer, f)
        print('replay data saved at %s' %demo_name)
    return new_buffer

def combine_batch(batch1, batch2):
    combined = []
    for i in range(len(batch1)):
        combined.append(torch.cat([batch1[i], batch2[i]]))
    return combined

def sample_combined_batch(buffer1, buffer2, batch_size):
    n_sample1 = int(batch_size/10)
    n_sample2 = int(batch_size - n_sample1)

    batch1 = buffer1.sample(n_sample1)
    batch2 = buffer2.sample(n_sample2)
    minibatch = combine_batch(batch1, batch2)
    return minibatch
