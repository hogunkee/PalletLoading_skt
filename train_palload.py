import argparse
import datetime
import time
import os
import json

import torch
from utils import *

from learning import evaluate
from agent.D_SAC import D_SAC_Agent

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--reward", default='dense', type=str)
    parser.add_argument("--max_levels", default=3, type=int)
    ## learning ##
    parser.add_argument("--algorithm", default='D-PPO', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--buff_size", default=1e5, type=float)
    parser.add_argument("--total_episodes", default=5e5, type=float)
    parser.add_argument("--learn_start", default=0, type=float)
    parser.add_argument("--log_freq", default=250, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_name", default="1127_2048", type=str)
    parser.add_argument("--num_trials", default=25, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = False # True False #args.render
    discrete_block = True #args.discrete
    max_steps = args.max_steps
    resolution = args.resolution
    reward_type = args.reward
    max_levels = args.max_levels

    # evaluate configuration #
    evaluation = False # True #args.evaluate
    # model_path = os.path.join(f"results/models/{args.algorithm}/")
    model_path = f"results/models/{args.algorithm}/"
    model_name = args.model_name
    num_trials = args.num_trials
    show_q = False #args.show_q

    # heuristics #
    use_bound_mask = True
    use_floor_mask = True
    use_projection = True
    use_coordnconv = True

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    now = datetime.datetime.now()
    savename = "%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists(f"results/models/{args.algorithm}/"):
            os.makedirs(f"results/models/{args.algorithm}/")

        if not os.path.exists(f"results/config/{args.algorithm}/"):
            os.makedirs(f"results/config/{args.algorithm}/")
        with open(f"results/config/{args.algorithm}/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

        if not os.path.exists(f"results/board/{args.algorithm}/"):
            os.makedirs(f"results/board/{args.algorithm}/")
        log_dir = f"results/board/{args.algorithm}/"

    # wandb log #
    log_name = savename
    wandb_off = True # args.wandb_off
    if not (evaluation or wandb_off):
        wandb.init(project="SKT Palletizing")
        wandb.run.name = log_name
        wandb.config.update(args)
        wandb.run.save()

    if max_levels == 1:
        from environment.environment import Floor1 as FloorEnv
    else:
        from environment.environment import FloorN as FloorEnv

    env = FloorEnv(
        resolution=resolution, 
        num_steps=max_steps,
        num_preview=5,
        box_norm=True,
        render=render,
        discrete_block=discrete_block,
        max_levels=max_levels,
        show_q=show_q,
        reward_type=reward_type
    )

    # learning configuration #
    algo = args.algorithm
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_episodes = int(args.total_episodes)
    learn_start = int(args.learn_start)
    log_freq = args.log_freq
    double = args.double

    if args.algorithm == "DQN":
        from learning import DQNlearning as learning
    elif args.algorithm == "D-PPO":
        from learning import DPPOlearning as learning
    elif args.algorithm == "D-SAC":
        from learning import DSAClearning as learning
    elif args.algorithm == "D-TAC":
        from learning import DTAClearning as learning
    else:
        NotImplementedError(f"{algo} is not implemented.")

    if evaluation:
        critic_path = model_path + f"{model_name}_critic.pth"
        actor_path = model_path + f"{model_name}_actor.pth"
        model_path = model_path + f"{model_name}.pth" if algo=="DQN" else [critic_path, actor_path]
        evaluate(env=env, algo=algo, model_path=model_path, num_trials=num_trials,
                 show_q=show_q, resolution=resolution, max_levels=max_levels, print_info=True,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv)
    else:
        learning(env=env, savename=savename, log_dir=log_dir, learning_rate=learning_rate, 
                 batch_size=batch_size, buff_size=buff_size,
                 total_episodes=total_episodes, learn_start=learn_start, log_freq=log_freq, 
                 double=double, model_path=model_path, 
                 wandb_off=wandb_off, show_q=show_q, resolution=resolution, max_levels=max_levels,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv)
