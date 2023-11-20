import argparse
import datetime
import time
import os
import json

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
        print_info=False,
    ):
    if agent is None:
        agent = Agent(max_levels, resolution,
                      train=False, model_path=model_path)

    log_returns = []
    log_eplen = []
    log_pf = []

    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        q_mask = None

        if use_projection:
            p_projection = 1.0
        else:
            p_projection = 0.0

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

        packing_factor = state.sum() / np.ones_like(state).sum()
        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_pf.append(packing_factor)

        if print_info:
            print("EP{}".format(ne+1), end=" / ")
            print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
            print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
            print("mean reward:{0:.1f}".format(np.mean(log_returns)), end=" / ")
            print("mean eplen:{0:.1f}".format(np.mean(log_eplen)), end=" / ")
            print("mean packing factor:{0:.3f}".format(np.mean(log_pf)))

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
        continue_learning=False,
        model_path='',
        wandb_off=False,
        show_q=False,
        resolution=20,
        max_levels=1,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
    ):
    agent = Agent(max_levels, resolution, True,
                  learning_rate, model_path, continue_learning, double)

    replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    if continue_learning:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'))
        log_returns = numpy_log[0].tolist()
        log_loss = numpy_log[1].tolist()
        log_eplen = numpy_log[2].tolist()
        log_test_len = numpy_log[3].tolist()
        log_test_pf = numpy_log[4].tolist()
        log_step = numpy_log[5].tolist()
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_test_len = []
        log_test_pf = []
        log_step = []

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
            p_projection = 0.5
        else:
            p_projection = 0.0

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

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

            ## save transition to the replay buffer ##
            replay_buffer.add(state, block, action, next_state, next_block, reward, done)

            state, block = next_state, next_block

            if replay_buffer.size < learn_start:
                if done: break
                else: continue

            minibatch = replay_buffer.sample(batch_size)
            loss = agent.update_network(minibatch, tau)
            log_minibatchloss.append(loss)

            if done: break

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
                                         use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask, use_projection=False)
            
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
            print(" / Loss:{0:.5f}".format(log_mean_loss[-1]), end="")
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
    parser.add_argument("--max_levels", default=3, type=int)
    ## learning ##
    parser.add_argument("--algorithm", default='DQN', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--buff_size", default=1e5, type=float)
    parser.add_argument("--total_episodes", default=5e5, type=float)
    parser.add_argument("--learn_start", default=1e3, type=float)
    parser.add_argument("--update_freq", default=250, type=int)
    parser.add_argument("--log_freq", default=250, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--continue_learning", action="store_true")
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="####_####", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = False # True False #args.render
    discrete_block = True #args.discrete
    max_steps = args.max_steps
    resolution = 20 #args.resolution
    reward_type = args.reward
    max_levels = args.max_levels

    # evaluate configuration #
    evaluation = False # True False #args.evaluate
    model_path = os.path.join("results/models/FCDQN_%s.pth"%args.model_path)
    num_trials = args.num_trials
    show_q = True #args.show_q

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    now = datetime.datetime.now()
    savename = "FCDQN-L%d_%s_%s" % (max_levels, now.strftime("%m%d_%H%M"), reward_type)
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
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = args.double

    continue_learning = args.continue_learning

    use_bound_mask = True
    use_floor_mask = True
    use_projection = True

    if args.algorithm == "DQN":
        from agent.DQN import DQN_Agent as Agent
    elif args.algorithm == "Discrete-PPO":
        raise NotImplementedError
    elif args.algorithm == "Discrete-SAC":
        raise NotImplementedError
    
    if evaluation:
        evaluate(env=env, model_path=model_path, num_trials=num_trials,
                 show_q=show_q, resolution=resolution, max_levels=max_levels, print_info=True,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask, use_projection=use_projection)
    else:
        learning(env=env, savename=savename, learning_rate=learning_rate, 
                 batch_size=batch_size, buff_size=buff_size,
                 total_episodes=total_episodes, learn_start=learn_start, log_freq=log_freq, 
                 double=double, continue_learning=continue_learning, model_path=model_path, 
                 wandb_off=wandb_off, show_q=show_q, resolution=resolution, max_levels=max_levels,
                 use_bound_mask=use_bound_mask, use_floor_mask=use_floor_mask, use_projection=use_projection)
