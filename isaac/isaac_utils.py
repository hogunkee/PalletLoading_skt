import omni
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

import cv2
import math
from copy import deepcopy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

BOX_HEIGHT = 0.156

#criterion = nn.SmoothL1Loss(reduction='mean').cuda()
criterion = nn.MSELoss(reduction='mean').cuda()

class Renderer(object):
    def __init__(self, resolution, show_q=False):
        if show_q:
            raise NotImplementedError
        
        plot0 = plt.subplot2grid((1,3), (0,0))
        plot1 = plt.subplot2grid((1,3), (0,1))
        plot2 = plt.subplot2grid((1,3), (0,2))
        plot0.set_title('Previous state')
        plot1.set_title('Current state')
        plot2.set_title('Current block')
        self.plots = [plot0, plot1, plot2]

        for i, p in enumerate(self.plots):
            p.set_xticks([])
            p.set_yticks([])
            if i==0 or i==1: p.axis('off')
        plt.show(block=False)

        self.resolution = resolution
        self.show_q = show_q

    def render_current_state(self, state, next_blocks, previous_state, q_value=None, box=None):
        next_blocks = [next_blocks]
        if len(np.shape(state)) == 3:
            state = state[0]
            previous_state = previous_state[0]

        b_gray = 0.85
        
        pad = int(0.1 * self.resolution)
        state_pad = np.ones([int(1.2*self.resolution), int(1.2*self.resolution), 3]) * b_gray
        pad_mask = np.ones(state_pad.shape[:2])
        pad_mask[pad:-pad, pad:-pad] = 0

        pre_state_pad = state_pad.copy()
        # y, x = np.where(previous_state==0)
        # pre_state_pad[y + pad, x + pad] = (1, 1, 1)
        # y, x = np.where(previous_state==1)
        # pre_state_pad[y + pad, x + pad] = (0, 0, 0)
        for y in range(np.shape(previous_state)[0]):
            for x in range(np.shape(previous_state)[1]):
                gray_ = 1.0 - previous_state[y,x]
                pre_state_pad[y + pad, x + pad] = (gray_, gray_, gray_)
        y, x = np.where(previous_state>1)
        pre_state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(np.all(pre_state_pad!=[b_gray,b_gray,b_gray], axis=-1) & (pad_mask==1))
        pre_state_pad[y, x] = (1, 0, 0)
        pre_state_pad = np.flip(pre_state_pad, axis=0)

        self.plots[0].imshow(pre_state_pad)

        if self.show_q:
            raise NotImplementedError

        state_pad = np.ones([int(1.2*self.resolution), int(1.2*self.resolution), 3]) * b_gray
        # y, x = np.where(state==0)
        # state_pad[y + pad, x + pad] = (1, 1, 1)
        # y, x = np.where(state==1)
        # state_pad[y + pad, x + pad] = (0, 0, 0)
        for y in range(np.shape(state)[0]):
            for x in range(np.shape(state)[1]):
                gray_ = 1.0 - state[y,x]
                state_pad[y + pad, x + pad] = (gray_, gray_, gray_)
        if box is not None:
            min_y, max_y, min_x, max_x = np.array(box) + pad
            state_pad[min_y: max_y, min_x: max_x] = (0, 0, 1)

        y, x = np.where(state>1)
        state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(np.all(state_pad!=[b_gray,b_gray,b_gray], axis=-1) & (pad_mask==1))
        state_pad[y, x] = (1, 0, 0)
        state_pad = np.flip(state_pad, axis=0)
        self.plots[1].imshow(state_pad)

        block_figures = None
            
        for i, b in enumerate(next_blocks):
            cy, cx = int(0.6*self.resolution), int(0.6*self.resolution)
            b = np.round(np.array(b) * self.resolution).astype(int)
            if len(b) > 2: b = b[:2]
            by, bx = b
            min_y, max_y = math.floor(cy-by/2), math.floor(cy+by/2)
            min_x, max_x = math.floor(cx-bx/2), math.floor(cx+bx/2)

            block_fig = np.ones_like(state_pad)
            block_fig[max(min_y,0): max_y, max(min_x,0): max_x] = [0, 0, 0]

            if i==0:
                self.plots[2].imshow(block_fig)
            else:
                if block_figures is None:
                    block_figures = block_fig
                else:
                    block_figures = np.concatenate([block_figures, block_fig], axis=1)
        plt.draw()
        # plt.pause(0.01)
        plt.pause(1)    

def box_info(prim_path):
    bbox_cache = create_bbox_cache()
    bounds = compute_aabb(bbox_cache=bbox_cache, prim_path=prim_path)
    return bounds
    

# def get_bin_info(bin_obj):
#     '''
#     Returns world pose, width, height, depth of the object given the name of
#     the object
#     '''
#     p, q = bin_obj.get_world_pose()
#     bounds = box_info(bin_obj.prim_path)
#     min_x, min_y, min_z = bounds[:3]
#     max_x, max_y, max_z = bounds[3:]
#     width = max_x - min_x
#     height = max_y - min_y
#     depth = max_z - min_z
    
#     return {
#         "position": p,
#         "rotation": q,
#         "width": width,
#         "height": height,
#         "depth": depth
#     }
def bbox_info(prim_path):
    # Calculate the bounds of the prim
    bb_cache = create_bbox_cache()
    prim = get_prim_at_path(prim_path)
    bbox3d_gf = bb_cache.ComputeLocalBound(prim)
    
    # Apply transformation to local frame
    prim_tf_gf = omni.usd.get_world_transform_matrix(prim)
    bbox3d_gf.Transform(prim_tf_gf)
    range_size = np.array(bbox3d_gf.GetRange().GetSize())
    return range_size

def scaled_bbox_info(prim_path, scale):
    range_size = bbox_info(prim_path)
    scale = np.array(scale)
    return range_size * scale
    
def get_bin_info(bin_obj):
    '''
    Returns world pose, width, height, depth of the object given the name of
    the object
    '''
    p, q = bin_obj.get_world_pose()
    scale = bin_obj.get_local_scale()
    bounds = scaled_bbox_info(bin_obj.prim_path, scale)
    width = bounds[0]
    height = bounds[1]
    depth = bounds[2]
    
    return {
        "position": p,
        "rotation": q,
        "width": width,
        "height": height,
        "depth": depth
    }
    
    
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
    next_qmask = minibatch[4]
    actions = minibatch[5].type(torch.long)
    rewards = minibatch[6]
    not_done = minibatch[7]
    batch_size = state.size()[0]

    def get_a_prime():
        next_q = FCQ(next_state, next_block)
        next_q *= next_qmask

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

def get_box_level(pose_list):
    level_list, eps = [], 0.02
    for pose in pose_list:
        current_level = int((pose[2]+eps)/BOX_HEIGHT)
        #assert 0 <=current_level < self.max_level
        level_list.append(current_level)
    return level_list

def check_stability(input_list, output_list,
                    min_pose_error=0.03, min_quat_error=10.0, print_info=False):
    input_pose = input_list['poses']
    input_quat = input_list['quats']
    output_pose = output_list['poses']
    output_quat = output_list['quats']
    n_boxes = len(input_pose)

    input_level = get_box_level(input_pose)
    output_level = get_box_level(output_pose)

    stability_ = True

    # conditions for position
    moved_boxes = []
    for i in range(n_boxes):
        dist_ = [(a-b)**2 for (a, b) in zip(
                    input_pose[i][:2], output_pose[i][:2]
                )]
        dist_ = math.sqrt(sum(dist_))

        rot_input_ = Rotation.from_quat(input_quat[i])
        euler_input_ = rot_input_.as_euler('xyz', degrees=True)
        rot_output_ = Rotation.from_quat(output_quat[i])
        euler_output_ = rot_output_.as_euler('xyz', degrees=True)
        euler_ = max([
            abs(euler_input_[0]-euler_output_[0]),
            abs(euler_input_[1]-euler_output_[1]),
            abs(euler_input_[2]-euler_output_[2]),
        ])
        #euler_ = abs(euler_input_[2]-euler_output_[2])
        euler_ = min(euler_, 180-euler_)

        # if dist_ > min_pose_error or input_level[i] != output_level[i] or euler_ > min_quat_error:
        #     stability_ = False
        #     moved_boxes.append(i)

        if dist_ > min_pose_error:
            stability_ = False
            moved_boxes.append(i)

            # if dist_ > min_pose_error: print("D:", i, dist_)
            # if input_level[i] != output_level[i]: print("L:", i, input_level[i], output_level[i])
            # if euler_ > min_quat_error: print("Q:", i, euler_input_[0]-euler_output_[0], euler_input_[1]-euler_output_[1], euler_input_[2]-euler_output_[2])

    if print_info:
        if len(moved_boxes) == 1:
            print("  * [Unstable] Box[{}] is Moved.".format(moved_boxes))
        elif len(moved_boxes) >= 2:
            print("  * [Unstable] Box[{}] are Moved.".format(moved_boxes))

    return stability_