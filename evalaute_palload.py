import argparse
import datetime
import time
import os
import json
#import skfmm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from utils import *
from replay_buffer import ReplayBuffer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

#crop_min = 0 #22 #45 #9 #19 #11 #13
#crop_max = 64 #220 #440 #88 #78 #54 #52


crop_min = 0 #22 #45 #9 #19 #11 #13
def get_action(env, fc_qnet, state, block, epsilon, crop_min=0, crop_max=64, pre_action=None, with_q=False, deterministic=True):
    #crop_min_y = 
    if np.random.random() < epsilon:
        action = [np.random.choice([0, 1]), np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max)]
        if with_q:
            state_tensor = torch.FloatTensor([state]).cuda()
            #state_tensor = state_tensor[:, None, :, :]
            block_tensor = torch.FloatTensor([block]).cuda()
            q_value = fc_qnet(state_tensor, block_tensor)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.ones_like(q_raw) * q_raw.min()
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_tensor = torch.FloatTensor([state]).cuda()
        #state_tensor = state_tensor[:, None, :, :]
        block_tensor = torch.FloatTensor([block]).cuda()
        q_value = fc_qnet(state_tensor, block_tensor)
        q_raw = q_value[0].detach().cpu().numpy()
        q = np.ones_like(q_raw) * q_raw.min()
        q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
        # avoid redundant motion #
        #if pre_action is not None:
        #    q[pre_action[0], pre_action[1], pre_action[2]] = q.min()
        # image coordinate #

        #deterministic = False
        if deterministic:
            aidx_y = q.max(0).max(1).argmax()
            aidx_x = q.max(0).max(0).argmax()
            aidx_th = q.argmax(0)[aidx_y, aidx_x]
        else:
            soft_tmp = 1e-1
            n_th, n_y, n_x = q.shape
            q_probs = q.reshape((-1,))
            q_probs = np.exp((q_probs-q_probs.max())/soft_tmp)
            q_probs = q_probs / q_probs.sum()

            aidx = np.random.choice(len(q_probs), 1, p=q_probs)[0]
            aidx_th = aidx // (n_y*n_x)
            aidx_y = (aidx % (n_y*n_x)) // n_x
            aidx_x = (aidx % (n_y*n_x)) % n_x
        action = [aidx_th, aidx_y, aidx_x]

    if with_q:
        return action, q
    else:
        return action


def evaluate(env, model_path='', num_trials=10, b1=0.1, b2=0.1, show_q=False, n_hidden=16,
             resolution=64, max_levels=1):
    FCQ = FCQNet(2, max_levels).cuda()
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))
    FCQ.eval()

    log_returns = []
    log_eplen = []
    log_pf = []

    pre_action = None

    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        pre_action = None
        for t_step in range(env.num_steps):
            ep_len += 1
            action, q_map = get_action(env, FCQ, state, block,
                                       epsilon=0.0, crop_min=0, crop_max=resolution,
                                       pre_action=pre_action, with_q=True, deterministic=True)
            if show_q:
                env.q_value = q_map
            obs, reward, done = env.step(action)

            next_state, next_block = obs
            if len(next_state.shape)==2:
                next_state = next_state[np.newaxis, :, :]
            episode_reward += reward

            if done:
                break
            else:
                state = next_state
                block = next_block
                pre_action = action

        packing_factor = state.sum() / np.ones_like(state).sum()
        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_pf.append(packing_factor)

        print("EP{}".format(ne+1), end=" / ")
        print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
        print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
        print("mean reward:{0:.1f}".format(np.mean(log_returns)), end=" / ")
        print("mean eplen:{0:.1f}".format(np.mean(log_eplen)), end=" / ")
        print("mean packing factor:{0:.3f}".format(np.mean(log_pf)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--b1", default=0.10, type=float)
    parser.add_argument("--b2", default=0.25, type=float)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--reward", default='dense_v2', type=str)
    parser.add_argument("--max_levels", default=1, type=int)
    ## learning ##
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--buff_size", default=1e5, type=float)
    parser.add_argument("--total_episodes", default=2e5, type=float)
    parser.add_argument("--learn_start", default=1e3, type=float)
    parser.add_argument("--update_freq", default=250, type=int)
    parser.add_argument("--log_freq", default=250, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    ## Evaluate ##
    parser.add_argument("--model_path", default="113_2302", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = False
    b1 = args.b1
    b2 = args.b2
    discrete_block = True #args.discrete
    max_steps = args.max_steps
    resolution = 20 #args.resolution
    reward_type = args.reward
    max_levels = args.max_levels

    # evaluate configuration #
    model_list = [f for f in os.listdir("results/models") if f.endswith(".pth")]
    model_name = [f for f in model_list if args.model_path in f]
    if len(model_name)==1:
        model_name = model_name[0]
    model_path = os.path.join("results/models/%s"%model_name)
    #model_path = os.path.join("results/models/FCDQN_%s.pth"%args.model_path)
    num_trials = args.num_trials
    show_q = True# args.show_q

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if max_levels == 1:
        from environment.environment import Floor1 as FloorEnv
    else:
        from environment.environment import FloorN as FloorEnv

    env = FloorEnv(
        resolution=resolution, 
        num_steps=max_steps,
        num_preview=5,
        box_norm=True,
        action_norm=False,
        render=render,
        block_size_min=b1,
        block_size_max=b2,
        discrete_block=discrete_block,
        max_levels=max_levels,
        show_q=show_q,
        reward_type=reward_type
    )

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_episodes = int(args.total_episodes)
    learn_start = int(args.learn_start)
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = args.double
    augmentation = args.augmentation

    #half = args.half
    #small = args.small
    continue_learning = args.continue_learning
    
    n_hidden = 16
    from models import FCQResNetSmallV1113 as FCQNet

    evaluate(env=env, model_path=model_path, num_trials=num_trials, b1=b1, b2=b2, 
             show_q=show_q, n_hidden=n_hidden, resolution=resolution, max_levels=max_levels)
