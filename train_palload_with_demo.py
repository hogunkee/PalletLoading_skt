import argparse
import datetime
import time
import os
import json
import pickle
from copy import deepcopy

import torch
from utils import *
from replay_buffer import ReplayBuffer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb


def evaluate(
        env,
        agent=None,
        model_path='',
        num_trials=10,
        show_q=False,
        resolution=20,
        max_levels=1,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
        use_coordnconv=False,
        use_terminal_reward=False,
        print_info=False,
    ):
    if agent is None:
        agent = Agent(max_levels, resolution, train=False,
                      model_path=model_path, use_coordnconv=use_coordnconv)

    log_returns = []
    log_eplen = []
    log_pf = []

    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        q_mask = None

        if use_projection:
            p_projection = 1.000
        else:
            p_projection = 0.000

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        for _ in range(env.num_steps):
            ep_len += 1

            if use_bound_mask:
                q_mask = generate_bound_mask(state, block)
            if use_floor_mask:
                q_mask = generate_floor_mask(state, block, q_mask)

            action, q_map = agent.get_action(state, block,
                                             with_q=True, deterministic=True,
                                             q_mask=q_mask, p_project=p_projection)

            if show_q:
                env.q_value = q_map

            obs, reward, done = env.step(action)

            state, block = obs
            if len(state.shape)==2:
                state = state[np.newaxis, :, :]
            episode_reward += reward

            if done: break

        if use_terminal_reward:
            terminal_reward = env.get_terminal_reward()
            episode_reward += terminal_reward

        packing_factor = state.sum() / np.ones_like(state).sum()
        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_pf.append(packing_factor)

        if print_info:
            print("EP{}".format(ne+1), end=" / ")
            print("Current: R{:.2f}, B{:.2f}, P{:.3f}".format(log_returns[-1],log_eplen[-1],log_pf[-1]), end=" / ")
            print("Mean: R{:.2f}, B{:.2f}, P{:.3f}".format(np.mean(log_returns),np.mean(log_eplen),np.mean(log_pf)))

    if print_info:
        print()
        print("="*80)
        print("Evaluation Done.")
        print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
        print("Mean episode length: {}".format(np.mean(log_eplen)))

    return np.mean(log_eplen), np.mean(log_pf)


def learning(
        env,
        savename,
        learning_rate=3e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_episodes=1e6,
        learn_start=1e4,
        log_freq=1e3,
        tau=1e-3,
        double=True,
        model_path='',
        wandb_off=False,
        show_q=False,
        resolution=20,
        max_levels=1,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
        use_coordnconv=False,
        use_terminal_reward=False,
    ):
    agent = Agent(max_levels, resolution, True,
                  learning_rate, model_path, double, use_coordnconv)

    replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    demo_filename = 'results/replay/replay_1.pkl'
    with open(demo_filename, 'rb') as f:
        demo_buffer = pickle.load(f)

    log_returns, log_loss, log_eplen, log_test_len, log_test_pf, log_step = [], [], [], [], [], []

    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")


    st = time.time()

    max_return = -1e7
    count_steps = 0

    for ne in range(total_episodes):
        ep_len = 0
        episode_reward = 0.
        log_minibatchloss = []

        q_mask = None

        if use_projection:
            p_projection = 0.700
        else:
            p_projection = 0.000

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        trajectories = []
        for _ in range(env.num_steps):
            count_steps += 1
            ep_len += 1

            if use_bound_mask:
                q_mask = generate_bound_mask(state, block)

            if use_floor_mask:
                q_mask = generate_floor_mask(state, block, q_mask)

            action, q_map = agent.get_action(state, block,
                                             with_q=True, deterministic=False,
                                             q_mask=q_mask, p_project=p_projection)
            
            if show_q:
                env.q_value = q_map

            obs, reward, done = env.step(action)

            next_state, next_block = obs
            if len(next_state.shape)==2:
                next_state = next_state[np.newaxis, :, :]
            episode_reward += reward

            next_q_mask = np.ones((2,resolution,resolution))
            if use_bound_mask:
                next_q_mask = generate_bound_mask(next_state, next_block)
            if use_floor_mask:
                next_q_mask = generate_floor_mask(next_state, next_block, next_q_mask)

            ## save transition to the replay buffer ##
            trajectory = deepcopy([state, block, action, next_state, next_block, next_q_mask, reward, done])
            trajectories.append(trajectory)

            state, block = next_state, next_block

            if replay_buffer.size < learn_start:
                if done: break
                else: continue

            minibatch_replay = replay_buffer.sample(batch_size//2)
            minibatch_demo = replay_buffer.sample(batch_size//2)
            minibatch = combine_batch(minibatch_replay, minibatch_demo)

            loss = agent.update_network(minibatch, tau)
            log_minibatchloss.append(loss)

            if done: break

        if use_terminal_reward:
            terminal_reward = env.get_terminal_reward()
            episode_reward += terminal_reward

            new_trajectories = recalculate_rewards(trajectories, terminal_reward, t_step+1)
            trajectories = new_trajectories

        for traj in trajectories:
            replay_buffer.add(*traj)

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_eplen.append(ep_len)

        eplog = {
            'Reward': episode_reward,
            'loss': np.mean(log_minibatchloss),
            'EP Len': ep_len,
        }
        if not wandb_off:
            wandb.log(eplog, count_steps)

        if ne % log_freq == 0:
            agent.train_on_off(train=False)
            test_len, test_pf = evaluate(env=env, agent=agent, num_trials=25,
                                         show_q=show_q, resolution=resolution, max_levels=max_levels,
                                         use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask, use_projection=True)
            
            log_test_len.append(test_len)
            log_test_pf.append(test_pf)
            log_step.append(count_steps)

            log_mean_returns = smoothing_log(log_returns, log_freq)
            log_mean_loss = smoothing_log(log_loss, log_freq)
            log_mean_eplen = smoothing_log(log_eplen, log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print("Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / Loss:{0:.3f}".format(log_mean_loss[-1]), end="")
            print(" / Train:{0:.1f}".format(log_mean_eplen[-1]), end="")
            print(" / Test:({:.1f}/{:.1f}%)".format(test_len,test_pf*100.0), end="")

            log_list = [
                log_returns,  # 0
                log_loss,  # 1
                log_eplen,  # 2
                log_test_len, 
                log_test_pf, 
                log_step, 
            ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                torch.save(agent.FCQ.state_dict(), 'results/models/%s.pth' % savename)
                print(" <- Highest Return. Saving the model.")
            else:
                torch.save(agent.FCQ.state_dict(), 'results/models/%s_last.pth' % savename)
                print("")

            agent.train_on_off(train=True)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--reward", default='dense', type=str)
    parser.add_argument("--max_levels", default=1, type=int)
    ## learning ##
    parser.add_argument("--algorithm", default='DQN', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--buff_size", default=1e5, type=float)
    parser.add_argument("--total_episodes", default=5e5, type=float)
    parser.add_argument("--learn_start", default=1000, type=float)
    parser.add_argument("--log_freq", default=250, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="####_####", type=str)
    parser.add_argument("--num_trials", default=25, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = False #args.render
    discrete_block = True #args.discrete
    max_steps = args.max_steps
    resolution = args.resolution
    reward_type = args.reward
    max_levels = args.max_levels

    # evaluate configuration #
    evaluation = False # True False #args.evaluate
    model_path = os.path.join("results/models/FCDQN_%s.pth"%args.model_path)
    num_trials = args.num_trials
    show_q = True #args.show_q

    # heuristics #
    use_bound_mask = False #True
    use_floor_mask = False #True
    use_projection = False #True
    use_coordnconv = True
    use_terminal_reward = False

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    now = datetime.datetime.now()
    savename = "FCDQN_%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

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
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_episodes = int(args.total_episodes)
    learn_start = int(args.learn_start)
    log_freq = args.log_freq
    double = args.double

    if args.algorithm == "DQN":
        from agent.DQN import DQN_Agent as Agent
    elif args.algorithm == "Discrete-PPO":
        raise NotImplementedError
    elif args.algorithm == "Discrete-SAC":
        raise NotImplementedError
    
    if evaluation:
        evaluate(env=env, model_path=model_path, num_trials=num_trials,
                 show_q=show_q, resolution=resolution, max_levels=max_levels, print_info=True,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv)
    else:
        learning(env=env, savename=savename, learning_rate=learning_rate, 
                 batch_size=batch_size, buff_size=buff_size,
                 total_episodes=total_episodes, learn_start=learn_start, log_freq=log_freq, 
                 double=double, model_path=model_path, wandb_off=wandb_off,
                 show_q=show_q, resolution=resolution, max_levels=max_levels,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv,
                 use_terminal_reward=use_terminal_reward)
