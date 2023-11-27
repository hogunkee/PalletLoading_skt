import math
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

    min_packed_ratio, empty_level = 0.80, -1
    for i in range(0,max_level-1):
        if np.mean(cum_state[i]) < min_packed_ratio:
            empty_level = i+1
            break

    if empty_level > 0:
        for level_limit in range(empty_level, max_level):
            if pre_mask is None:
                mask = np.ones((2,resolution,resolution))
            else:
                mask = np.copy(pre_mask)

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

def cg(f_Ax, b, actor, obs_batch, cg_iters=10, residual_tol=1e-10):
    """
    # https://github.com/openai/baslines/blob/master/baselines.common/cg.py
    conjugate gradient algorithm
    here, f_Ax is a function which computes matrix-vector product efficiently
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p, actor, obs_batch)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    # return a search direction by solving Ax = g, where g : gradient of loss, and A : Fisher information matrix
    return x


def fisher_vector_product(v, actor, obs_batch, cg_damping=1e-2):
    # efficient Hessian-vector product
    # in our implementation, Hessian just corresponds to Fisher information matrix I
    v.detach()
    kl = torch.mean(kl_div(actor=actor, old_actor=actor, obs_batch=obs_batch))

    kl_grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])

    kl_grad_p = torch.sum(kl_grad * v)
    Iv = torch.autograd.grad(kl_grad_p, actor.parameters()) # product of Fisher information I and v
    Iv = flatten(Iv)

    return Iv + v * cg_damping


def backtracking_line_search(old_actor, actor, actor_loss, actor_loss_grad,
                             old_policy, params, maximal_step, max_kl,
                             adv, states, actions):
    backtrac_coef = 1.0
    alpha = 0.5
    beta = 0.5
    flag = False

    expected_improve = (actor_loss_grad * maximal_step).sum(0, keepdim=True)

    for i in range(10):
        new_params = params + backtrac_coef * maximal_step
        update_model(actor, new_params)

        new_actor_loss = surrogate_loss(actor, adv, states, old_policy.detach(), actions)

        loss_improve = new_actor_loss - actor_loss
        expected_improve *= backtrac_coef
        improve_condition = loss_improve / expected_improve

        kl = kl_div(actor=actor, old_actor=old_actor, obs_batch=states)
        kl = kl.mean()

        if kl < max_kl and improve_condition > alpha:
            flag = True
            break

        backtrac_coef *= beta

    if not flag:
        params = flat_params(old_actor)
        update_model(actor, params)


def kl_div(actor, old_actor, obs_batch):
    """
    Kullback-Leibler divergence between two action distributions ($\pi(\cdot \vert s ; \phi)$ and \pi(\cdot \vert s ; \phi_\text{old})$)
    we assume that both distributions are Gaussian with diagonal covariance matrices
    """
    mu, sigma = actor(obs_batch, with_sigma=True)

    mu_old, sigma_old = old_actor(obs_batch, with_sigma=True)
    mu_old = mu_old.detach()
    sigma_old = sigma_old.detach()

    kl = torch.log(sigma / sigma_old) + (sigma_old**2 + (mu_old - mu)**2) / (2.0 * sigma**2) - 0.5

    # return a batch [kl_0, ... , kl_{N-1}]^T
    # shape : [batch_size, 1]
    kl_batch = torch.sum(kl, dim=1, keepdim=True)

    return kl_batch


def flatten(hess):
    flat_hess = []
    for hessian in hess:
        flat_hess.append(hessian.contiguous().view(-1))
    flat_hess = torch.cat(flat_hess).data
    return flat_hess


def surrogate_loss(actor, adv, states, old_log_probs, actions):

    log_probs = actor.log_prob(states, actions)
    loss = torch.exp(log_probs - old_log_probs) * adv
    loss = loss.mean()

    return loss


def flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        len_params = len(params.view(-1))
        new_param = new_params[index: index + len_params]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += len_params


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

