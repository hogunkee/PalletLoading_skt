import argparse
import datetime
import time
import os
import json
import pickle
import torch
from utils import *


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb


def evaluate(
        env,
        agent=None,
        model_path='',
        config=None,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
        use_coordnconv=False,
        print_result=False,
    ):
    if agent is None:
        agent = Agent(False, model_path, use_coordnconv, config)

    log_returns = []
    log_eplen = []
    log_pf = []

    for ne in range(int(config.num_trials)):
        ep_len = 0
        episode_reward = 0.

        if use_projection: p_projection = 1.00
        else: p_projection = 0.00

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        for _ in range(env.num_steps):
            ep_len += 1

            q_mask = np.ones((2,resolution,resolution))
            if use_bound_mask:
                q_mask = generate_bound_mask(state, block)
            if use_floor_mask:
                q_mask = generate_floor_mask(state, block, q_mask)

            action, action_info = agent.get_action(state, block, q_mask,
                                                   with_q=config.show_q,
                                                   deterministic=True,
                                                   p_project=p_projection)
            if config.show_q: env.q_value = action_info

            obs, reward, done = env.step(action)

            state, block = obs
            if len(state.shape)==2:
                state = state[np.newaxis, :, :]
            episode_reward += reward
            if done: break

        packing_factor = state.sum() / np.ones_like(state).sum()
        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_pf.append(packing_factor)

        if print_result:
            print("EP{}".format(ne+1), end=" / ")
            print("Current: R{:.2f}, B{:.2f}, P{:.3f}".format(log_returns[-1],log_eplen[-1],log_pf[-1]), end=" / ")
            print("Mean: R{:.2f}, B{:.2f}, P{:.3f}".format(np.mean(log_returns),np.mean(log_eplen),np.mean(log_pf)))

    if print_result:
        print()
        print("="*80)
        print("Evaluation Done.")
        print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
        print("Mean episode length: {}".format(np.mean(log_eplen)))

    return np.mean(log_eplen), np.mean(log_pf)


def learning(
        env,
        save_name,
        model_path="",
        learning_type="",
        config=None,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
        use_coordnconv=False,
        use_full_demos=False,
    ):
    agent = Agent(True, model_path, use_coordnconv, config)

    if learning_type == "off_policy":
        from replay_buffer import OffReplayBuffer as ReplayBuffer
        replay_buffer = ReplayBuffer(
            [config.max_levels, config.resolution, config.resolution],
            2, dim_action=3, max_size=int(config.buff_size)
        )
    elif learning_type == "on_policy":
        from replay_buffer import OnReplayBuffer as ReplayBuffer
        replay_buffer = ReplayBuffer(
            dimS=[config.max_levels, config.resolution, config.resolution],
            dimA=3, gamma=config.gamma, lam=config.lam, lim=int(config.buff_size)
        )

    if use_full_demos and learning_type == "off_policy":
        demo_base = 'replay_0_B'
        demo_name = demo_base
        if use_floor_mask: demo_name += "_F"
        if config.max_levels > 1: demo_name += "_L{}".format(config.max_levels)
        demo_filename = 'demo/replay/{}.pkl'.format(demo_name)

        if os.path.exists(demo_filename):
            with open(demo_filename, 'rb') as f:
                demo_buffer = pickle.load(f)
        else:
            demo_initname = 'demo/replay/{}.pkl'.format(demo_base)
            with open(demo_initname, 'rb') as f:
                demo_buffer = pickle.load(f)
            demo_buffer= convert_full_demonstrations(demo_buffer, ReplayBuffer, demo_filename,
                                                     use_floor_mask=use_floor_mask, max_levels=config.max_levels, resolution=config.resolution)
    else:
        demo_buffer = None

    log_train_returns, log_train_loss, log_train_len, log_train_pf = [], [], [], []
    log_test_len, log_test_pf, log_test_step = [], [], []

    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    st = time.time()

    max_return = -1e7
    count_steps = 0

    for ne in range(int(config.total_episodes)):
        ep_len = 0
        episode_reward = 0.
        log_minibatchloss = []

        if use_projection: p_projection = 0.70
        else: p_projection = 0.00

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        q_mask = np.ones((2,resolution,resolution))
        if use_bound_mask:
            q_mask = generate_bound_mask(state, block)
        if use_floor_mask:
            q_mask = generate_floor_mask(state, block, q_mask)

        for t in range(env.num_steps):
            count_steps += 1
            ep_len += 1

            action, action_info = agent.get_action(state, block, q_mask,
                                                   soft_tmp=config.soft_tmp,
                                                   with_q=config.show_q,
                                                   deterministic=False,
                                                   p_project=p_projection)
            if learning_type == "off_policy":
                q_map = action_info
                if config.show_q: env.q_value = q_map
            elif learning_type == "on_policy":
                action_probs, state_value = action_info
            
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

            if learning_type == "off_policy":
                replay_buffer.add(state, block, q_mask, action, next_state, next_block, next_q_mask, reward, done)

                if demo_buffer is None:
                    minibatch = replay_buffer.sample(int(config.batch_size))
                else:
                    minibatch = sample_combined_batch(replay_buffer, demo_buffer, int(config.batch_size))
                loss = agent.update_network(minibatch)
                log_minibatchloss.append(loss)

            elif learning_type == "on_policy":
                replay_buffer.add(state, block, action, reward, state_value, action_probs, q_mask)
    
                if (t == env.num_steps-1 or replay_buffer._size == int(config.buff_size)):
                    v_last = agent.get_statevalue(next_state, next_block, next_q_mask)
                    replay_buffer.compute_values(v_last)
                elif done:
                    v_last = torch.zeros([2, resolution, resolution]).cuda()
                    replay_buffer.compute_values(v_last)

                if replay_buffer._size == int(config.buff_size):
                    fullbatch = replay_buffer.load()
                    loss = agent.update_network(fullbatch)
                    log_minibatchloss.append(loss)
                    replay_buffer.reset()
                    break
            
            state, block, q_mask = next_state, next_block, next_q_mask
            if done: break

        ep_pf = np.sum(next_state) / np.sum(np.ones_like(next_state))

        ep_loss = 0.0
        if len(log_minibatchloss) > 0:
            ep_loss = np.mean(log_minibatchloss)
        elif len(log_train_loss) > 0:
            ep_loss = log_train_loss[-1]

        log_train_returns.append(episode_reward)
        log_train_loss.append(ep_loss)
        log_train_len.append(ep_len)
        log_train_pf.append(ep_pf)

        eplog = {
            'Reward': episode_reward,
            'loss': ep_loss,
            'EP Len': ep_len,
        }
        if not wandb_off:
            wandb.log(eplog, count_steps)

        print_log = False
        if learning_type == "off_policy":
            print_log = ne % config.log_freq == 0 and ne > 0
        elif learning_type == "on_policy":
            #print_log = replay_buffer._size == 0
            log_interval = max(int(config.buff_size), 2000)
            print_log = count_steps % int(log_interval) == 0

        if print_log:
            agent.train_on_off(train=False)
            test_len, test_pf = evaluate(env=env, agent=agent, config=config,
                                         use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                                         use_projection=use_projection, use_coordnconv=use_coordnconv)
            
            log_test_len.append(test_len)
            log_test_pf.append(test_pf)
            log_test_step.append(count_steps)

            log_mean_returns = smoothing_log(log_train_returns, config.log_freq)
            log_mean_loss = smoothing_log(log_train_loss, config.log_freq)
            log_mean_eplen = smoothing_log(log_train_len, config.log_freq)
            log_mean_eppf = smoothing_log(log_train_pf, config.log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print("Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / Loss:{0:.3f}".format(log_mean_loss[-1]), end="")
            print(" / Train:({:.1f}/{:.1f}%)".format(log_mean_eplen[-1],log_mean_eppf[-1]*100.0), end="")
            print(" / Test:({:.1f}/{:.1f}%)".format(test_len,test_pf*100.0), end="")

            log_list = [
                log_train_returns,
                log_train_len,
                log_train_pf,
                log_train_loss,
                log_test_len, 
                log_test_pf, 
                log_test_step, 
            ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %save_name, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                agent.save_network(save_name, "best")
                print(" <- Highest Return. Saving the model.", end="")
            
            agent.save_network(save_name, "last")
            print("")

            agent.train_on_off(train=True)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## Env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discrete", action="store_false") # default: True
    parser.add_argument("--max_steps", default=70, type=int)
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--reward", default='dense', type=str)
    parser.add_argument("--max_levels", default=5, type=int)
    ## Learning ##
    parser.add_argument("--algorithm", default='D-PPO', type=str, help='[DQN, D-PPO, D-TAC]')
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--buff_size", default=512, type=float)
    parser.add_argument("--tau", default=1e-3, type=float)
    parser.add_argument("--grad_clip", default=1e0, type=float)
    parser.add_argument("--soft_tmp", default=1e-1, type=float)
    parser.add_argument("--total_episodes", default=5e5, type=float)
    parser.add_argument("--log_freq", default=250, type=int)
    ## For DQN ##
    parser.add_argument("--do_double", action="store_false") # default: True
    ## For Discrete-PPO ##
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--num_updates", default=1, type=int)
    ## For Discrete-TAC ##
    parser.add_argument("--q_prime", default=1.20, type=float)
    parser.add_argument("--target_entropy_ratio", default=0.98, type=float)
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="1204_1750_best", type=str)
    parser.add_argument("--num_trials", default=25, type=int)
    ## Stack Condtions ##
    parser.add_argument("--use_bound_mask", action="store_true")
    parser.add_argument("--use_floor_mask", action="store_true")
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--use_coordnconv", action="store_true")
    parser.add_argument("--use_full_demos", action="store_true")
    ## ETC ##
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = True # True False #args.render
    discrete_block = args.discrete
    max_steps = args.max_steps
    resolution = args.resolution
    reward_type = args.reward
    max_levels = args.max_levels

    # evaluate configuration #
    evaluation = True #args.evaluate
    num_trials = args.num_trials
    show_q = args.show_q

    # heuristics #
    use_bound_mask = True # args.use_bound_mask
    use_floor_mask = True # args.use_floor_mask
    use_projection = True # args.use_projection
    use_coordnconv = True # args.use_coordnconv
    use_full_demos = True # args.use_full_demos

    args.model_path = "1204_1745_best"

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
        render=render,
        discrete_block=discrete_block,
        max_levels=max_levels,
        show_q=show_q,
        reward_type=reward_type
    )

    now = datetime.datetime.now()

    if args.algorithm == "DQN":
        from agent.DQN import DQN_Agent as Agent
        learning_type = "off_policy"
        model_path = os.path.join("results/models/DQN_%s.pth"%args.model_path)
        save_name = "DQN_%s" % (now.strftime("%m%d_%H%M"))

    elif args.algorithm == "D-PPO":
        from agent.D_PPO import DiscretePPO_Agent as Agent
        learning_type = "on_policy"
        args.show_q = False
        model_path = [
            os.path.join("results/models/DPPO_%s_critic.pth"%args.model_path),
            os.path.join("results/models/DPPO_%s_actor.pth"%args.model_path),
        ]
        save_name = "DPPO_%s" % (now.strftime("%m%d_%H%M"))

    elif args.algorithm == "D-TAC":
        from agent.D_TAC import DiscreteTAC_Agent as Agent
        learning_type = "off_policy"
        model_path = [
            os.path.join("results/models/DTAC_%s_critic.pth"%args.model_path),
            os.path.join("results/models/DTAC_%s_actor.pth"%args.model_path),
        ]
        save_name = "DTAC_%s" % (now.strftime("%m%d_%H%M"))
    

    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % save_name, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

    # wandb log #
    wandb_off = True # args.wandb_off
    if not (evaluation or wandb_off):
        wandb.init(project="SKT Palletizing")
        wandb.run.name = save_name
        wandb.config.update(args)
        wandb.run.save()
    
    if evaluation:
        evaluate(env=env, model_path=model_path, config=args, print_result=True,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv)
    else:
        learning(env=env, save_name=save_name, model_path=model_path,
                 learning_type=learning_type, config=args,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask,
                 use_projection=use_projection, use_coordnconv=use_coordnconv,
                 use_full_demos=use_full_demos)
