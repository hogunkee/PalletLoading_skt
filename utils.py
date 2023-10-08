import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn

criterion = nn.SmoothL1Loss(reduction='mean').cuda()

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
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
    #next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = FCQ(state)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    def get_a_prime_pixel():
        next_q = FCQ(next_state)
        next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    def get_a_prime():
        next_q = FCQ(next_state)
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    a_prime = get_a_prime()

    next_q_target = FCQ_target(next_state)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
    #next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    #q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## separate FCDQN Loss ##
def calculate_loss_separate(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_double_separate(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state, True)
    q_values = FCQ(state, True)
    next_q = FCQ(next_state, True)

    def get_a_prime_pixel(obj):
        next_q_chosen = next_q[torch.arange(batch_size), obj, :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    def get_a_prime(obj):
        next_q_obj = next_q[:, obj]
        aidx_x = next_q_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q_target_s_a_prime = next_q_target[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        #next_q_target_chosen = next_q_target[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]]
        #q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

## Constrained separate FCDQN ##
def calculate_loss_constrained(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0).view(-1)
    return loss, error

def calculate_loss_double_constrained(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state)
    q_values = FCQ(state)
    next_q = FCQ(next_state)

    def get_a_prime(obj):
        next_q_chosen = next_q[torch.arange(batch_size), 0, obj, :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        next_q_target_chosen = next_q_target[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]]
        q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0).view(-1)
    return loss, error

def calculate_loss_next_v(minibatch, FCQ, FCQ_target, n_blocks):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    # rewards = minibatch[3]
    # not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state) # bs x 2 x nb x 8 x h x w
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        y_target = next_q[torch.arange(batch_size), 0, o].mean([1,2,3])
        pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_next_q(minibatch, FCQ, FCQ_target, n_blocks):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    # rewards = minibatch[3]
    # not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state) # bs x 2 x nb x 8 x h x w
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        y_target = next_q[torch.arange(batch_size), 0, o].max(1)[0].max(1)[0].max(1)[0]
        pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error


## Cascade FCDQN v1 (step-by-step) ##
def calculate_loss_cascade_v1(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state, True)
    next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
    #next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards[:,0].unsqueeze(1) + gamma * not_done * next_q_max

    q_values = FCQ(state, True)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_cascade_v1(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ(next_state, True)
    next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    _, a_prime = next_q_chosen.max(1, True)

    next_q_target = FCQ_target(next_state, True)
    next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
    y_target = rewards[:,0].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state, True)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_cascade_v1(minibatch, FCQ, CQN, CQN_target, gamma=0.5, add=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    q1_map = FCQ(state, True)

    state_withq = torch.cat((state_im, goal_im, q1_map), 1)
    next_state_withq = torch.cat((next_state_im, goal_im, q1_map), 1)
    next_q2 = CQN_target(next_state_withq, True)
    q2_values = CQN(state_withq, True)

    if add:
        rewards = rewards[:,0] + rewards[:,1]
    else:
        rewards = rewards[:,1]

    next_q2_max = next_q2.max(1)[0].max(1)[0].max(1)[0]
    #next_q2_max = next_q2[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max

    pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_cascade_v1(minibatch, FCQ, CQN, CQN_target, gamma=0.5, add=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    q1_map = FCQ(state, True)
    next_state = torch.cat((next_state_im, goal_im), 1)
    next_q1_map = FCQ(next_state, True)

    state_withq = torch.cat((state_im, goal_im, q1_map), 1)
    next_state_withq = torch.cat((next_state_im, goal_im, q1_map), 1)
    next_q2_target = CQN_target(next_state_withq, True)
    q2_values = CQN(state_withq, True)

    def get_a_prime():
        next_q2 = CQN(next_state_withq, True)
        next_q2_chosen = next_q2[torch.arange(batch_size), :, actions[:,0], actions[:,1]]
        if add:
            next_q1_chosen = next_q1_map[torch.arange(batch_size), :, actions[:,0], actions[:,1]]
            next_q_chosen = next_q1_chosen + next_q2_chosen
            _, a_prime = next_q_chosen.max(1, True)
        else:
            _, a_prime = next_q2_chosen.max(1, True)
        return a_prime

    loss = []
    error = []
    if add:
        rewards = rewards[:, 0] + rewards[:, 1]
    else:
        rewards = rewards[:, 1]

    a_prime = get_a_prime()
    next_q2_target_chosen = next_q2_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
    y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime

    pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Cascade FCDQN v2 (end-to-end) ##
def calculate_loss_cascade_v2(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_double_cascade_v2(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state)
    q_values = FCQ(state)
    next_q = FCQ(next_state)

    def get_a_prime(obj):
        next_q_obj = next_q[:, obj]
        aidx_x = next_q_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q_target_s_a_prime = next_q_target[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error


## Cascade FCDQN v3 (pre-trained) ##
def normalize_q(q_map):
    q_mean = q_map.mean([-3,-2,-1], True)
    return (q_map - q_mean) / q_mean


def calculate_cascade_loss_cascade_v3(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5, output='', normalize=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    if normalize:
        q1_values = normalize_q(q1_values)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    if normalize:
        next_q1_values = normalize_q(next_q1_values)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_targets).max(1)[0].max(1)[0].max(1)[0]
    #next_q2_max = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    #next_qsum_max = (next_q1_values + next_q2_targets)[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    if output=='':
        rewards = rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max
        pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output=='addR':
        rewards = rewards[:,0] + rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max
        pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output=='addQ':
        rewards = rewards[:,0] + rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_qsum_max
        pred = (q1_values+q2_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_cascade_v3(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5, output='', normalize=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    if normalize:
        q1_values = normalize_q(q1_values)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    if normalize:
        next_q1_values = normalize_q(next_q1_values)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    def get_a_prime():
        next_q2 = CQN(next_state_goal_q, True)
        if output == '' or output == 'addR':
            next_q = next_q2
        elif output == 'addQ':
            next_q = next_q1_values + next_q2
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    if output=='':
        rewards = rewards[:, 1]
        a_prime = get_a_prime()
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_q2_target_chosen = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime
        pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]

    elif output=='addR':
        rewards = rewards[:, 0] + rewards[:, 1]
        a_prime = get_a_prime()
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_q2_target_chosen = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime
        pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
        
    elif output=='addQ':
        rewards = rewards[:, 0] + rewards[:, 1]
        a_prime = get_a_prime()
        next_qsum = next_q1_values + next_q2_targets
        qsum_target_s_a_prime = next_qsum[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_qsum_target_chosen = next_qsum[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # qsum_target_s_a_prime = next_qsum_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * qsum_target_s_a_prime
        pred = (q1_values+q2_values)[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]

    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## separate Cascade FCDQN ##
def calculate_cascade_loss_sppcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    n_blocks = 2
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q2_max

        pred = q2_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_sppcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    n_blocks = 2
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)
    next_q2 = CQN(next_state_goal_q, True)

    def get_a_prime(obj):
        next_q2_obj = next_q2[:, obj]
        aidx_x = next_q2_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q2_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q2_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q2_target_s_a_prime

        pred = q2_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Cascade FCDQN 3blocks (pre-trained) ##
def calculate_cascade_loss_cascade_3blocks(minibatch, FCQ, FCQ2, FCQ3, FCQ3_target, gamma=0.5, output=''):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)

    state_goal_q1 = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = FCQ2(state_goal_q1, True)

    state_goal_q2 = torch.cat((state_im, goal_im, q2_values), 1)
    q3_values = FCQ3(state_goal_q2, True)

    next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)

    next_state_goal_q1 = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_values = FCQ2(next_state_goal_q1, True)

    next_state_goal_q2 = torch.cat((next_state_im, goal_im, next_q2_values), 1)
    next_q3_targets = FCQ3_target(next_state_goal_q2, True)

    next_q3_max = next_q3_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_values + next_q3_targets).max(1)[0].max(1)[0].max(1)[0]

    if output == '':
        rewards = rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q3_max
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output == 'addR':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q3_max
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output == 'addQ':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_qsum_max
        pred = (q1_values + q2_values + q3_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


def calculate_cascade_loss_double_cascade_3blocks(minibatch, FCQ, FCQ2, FCQ3, FCQ3_target, gamma=0.5, output=''):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)

    state_goal_q1 = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = FCQ2(state_goal_q1, True)

    state_goal_q2 = torch.cat((state_im, goal_im, q2_values), 1)
    q3_values = FCQ3(state_goal_q2, True)

    next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)

    next_state_goal_q1 = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_values = FCQ2(next_state_goal_q1, True)

    next_state_goal_q2 = torch.cat((next_state_im, goal_im, next_q2_values), 1)
    next_q3_targets = FCQ3_target(next_state_goal_q2, True)

    def get_a_prime():
        next_q3 = FCQ3(next_state_goal_q2, True)
        if output == '' or output == 'addR':
            next_q = next_q3
        elif output == 'addQ':
            next_q = next_q1_values + next_q2_values + next_q3
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    if output == '':
        rewards = rewards[:, 2]
        a_prime = get_a_prime()
        q3_target_s_a_prime = next_q3_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * q3_target_s_a_prime
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    elif output == 'addR':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        a_prime = get_a_prime()
        q3_target_s_a_prime = next_q3_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * q3_target_s_a_prime
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    elif output == 'addQ':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        a_prime = get_a_prime()
        next_qsum = next_q1_values + next_q2_values + next_q3_targets
        qsum_target_s_a_prime = next_qsum[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * qsum_target_s_a_prime
        pred = (q1_values + q2_values + q3_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

## curr PCQN ##
def calculate_loss_curr_pcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q)

    next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_targets).max(1)[0].max(1)[0].max(1)[0]

    y_target = rewards + gamma * not_done * next_q2_max
    pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_curr_pcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q)

    def get_a_prime():
        next_q2 = CQN(next_state_goal_q)
        next_q = next_q2

        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    a_prime = get_a_prime()
    q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

    y_target = rewards + gamma * not_done * q2_target_s_a_prime
    pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
        
    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## curr REQN ##
def calculate_loss_curr_reqn(minibatch, CQN, CQN_target, goal_type, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    empty_q = torch.zeros(batch_size, 8, *goal_im.size()[2:])
    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1], empty_q), 1)
    else:
        state_goal = torch.cat((state_im, goal_im, empty_q), 1)
    q1_values = CQN(state_goal)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1], empty_q), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im, empty_q), 1)
    next_q1_values = CQN_target(next_state_goal)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q)

    next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_targets).max(1)[0].max(1)[0].max(1)[0]

    y_target = rewards + gamma * not_done * next_q2_max
    pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_curr_reqn(minibatch, CQN, CQN_target, goal_type, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    empty_q = torch.zeros(batch_size, 8, *goal_im.size()[2:])
    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1], empty_q), 1)
    else:
        state_goal = torch.cat((state_im, goal_im, empty_q), 1)
    q1_values = CQN(state_goal)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1], empty_q), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im, empty_q), 1)
    next_q1_values = CQN_target(next_state_goal)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q)

    def get_a_prime():
        next_q2 = CQN(next_state_goal_q)
        next_q = next_q2

        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    a_prime = get_a_prime()
    q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

    y_target = rewards + gamma * not_done * q2_target_s_a_prime
    pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
        
    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

