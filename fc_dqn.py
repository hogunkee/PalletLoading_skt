import argparse
import datetime
import time
import os
import json
import skfmm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from utils import *
from replay_buffer import ReplayBuffer
from environment import Floor1

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

crop_min = 4 #22 #45 #9 #19 #11 #13
crop_max = 62 #220 #440 #88 #78 #54 #52


def get_action(env, fc_qnet, state, block, epsilon, pre_action=None, with_q=False):
    #crop_min_y = 
    if np.random.random() < epsilon:
        action = [0, np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max)]
        if with_q:
            state_tensor = torch.FloatTensor([state]).cuda()
            #state_tensor = state_tensor[:, None, :, :]
            block_tensor = torch.FloatTensor([block]).cuda()
            q_value = fc_qnet(state_tensor, block_tensor)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_tensor = torch.FloatTensor([state]).cuda()
        #state_tensor = state_tensor[:, None, :, :]
        block_tensor = torch.FloatTensor([block]).cuda()
        q_value = fc_qnet(state_tensor, block_tensor)
        q_raw = q_value[0].detach().cpu().numpy()
        q = np.zeros_like(q_raw)
        q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
        # avoid redundant motion #
        if pre_action is not None:
            q[pre_action[0], pre_action[1], pre_action[2]] = q.min()
        # image coordinate #
        aidx_y = q.max(0).max(1).argmax()
        aidx_x = q.max(0).max(0).argmax()
        aidx_th = q.argmax(0)[aidx_y, aidx_x]
        action = [aidx_th, aidx_y, aidx_x]
        #action = [aidx_x, aidx_y, aidx_th]

    if with_q:
        return action, q
    else:
        return action


def evaluate(env, model_path='', num_trials=10, b1=0.1, b2=0.1, show_q=False, n_hidden=16,
             resolution=64):
    FCQ = FCQNet(1, 1).cuda()
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    log_returns = []
    log_eplen = []

    pre_action = None
    if show_q:
        plt.rc('axes', labelsize=8)
        plt.rc('font', size=8)
        plt.show()
        fig = plt.figure()
        fig.set_figheight(3)
        fig.set_figwidth(7)

        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax0.set_title('Goal')
        ax1.set_title('State')
        ax2.set_title('Q-value')

        fig2, ax = plt.subplots(2, 4)
        fig2.set_figwidth(10)
        for i in range(2):
            for j in range(4):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_title("%d\xb0" %((4*i+j)*45))

        s0 = deepcopy(state[0]).transpose([1,2,0])
        s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

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
            action, q_map = get_action(env, FCQ, state, block, epsilon=0.0, pre_action=pre_action, with_q=True)
            if show_q:
                s0 = deepcopy(state).transpose([1, 2, 0])
                s1 = deepcopy(goal).transpose([1, 2, 0])
                im0 = ax0.imshow(s1)
                s0[action[0], action[1]] = [1, 0, 0]
                # q_map = q_map[0]
                for i in range(2):
                    for j in range(4):
                        ax[i][j].imshow(q_map[4*i+j], vmin=-0.2, vmax=1.5)
                q_map = q_map.transpose([1,2,0]).max(2)
                # q_map[action[0], action[1]] = 1.5
                ax1.imshow(s0)
                ax2.imshow(q_map, vmin=-0.2, vmax=1.5)

                fig.canvas.draw()
                fig2.canvas.draw()

            obs, reward, done = env.step(action[1:])
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

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)

        print("EP{}".format(ne+1), end=" / ")
        print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
        print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
        print(" / mean reward:{0:.1f}".format(np.mean(log_returns)), end="")
        print(" / mean eplen:{0:.1f}".format(np.mean(log_eplen)), end="")

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))


def learning(
        env, 
        savename,
        learning_rate=1e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_episodes=1e6,
        learn_start=1e4,
        update_freq=100,
        log_freq=1e3,
        double=True,
        continue_learning=False,
        model_path='',
        wandb_off=False,
        b1=0.1,
        b2=0.1,
        show_q=False,
        n_hidden=16,
        augmentation=False,
        resolution=64,
        ):

    FCQ = FCQNet(1, 1).cuda()
    if continue_learning:
        FCQ.load_state_dict(torch.load(model_path))
    FCQ_target = FCQNet(1, 1).cuda()
    FCQ_target.load_state_dict(FCQ.state_dict())

    # criterion = nn.SmoothL1Loss(reduction=None).cuda()
    # criterion = nn.MSELoss(reduction='mean')
    #optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer([1, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    if double:
        calculate_loss = calculate_loss_double_fcdqn
    else:
        calculate_loss = calculate_loss_fcdqn

    if continue_learning:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'))
        log_returns = numpy_log[0].tolist()
        log_loss = numpy_log[1].tolist()
        log_eplen = numpy_log[2].tolist()
        log_epsilon = numpy_log[3].tolist()
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []

    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    plt.show(block=False)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)

    #lr_decay = 0.98
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    if len(log_epsilon) == 0:
        epsilon = 0.5 #1.0
        start_epsilon = 0.5
    else:
        epsilon = log_epsilon[-1]
        start_epsilon = log_epsilon[-1]
    min_epsilon = 0.1
    epsilon_decay = 0.98
    max_return = -100
    st = time.time()

    if show_q:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        s0 = deepcopy(state).transpose([1,2,0])
        s1 = deepcopy(goal).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    count_steps = 0
    learning_starts = False
    for ne in range(total_episodes):
        ep_len = 0
        episode_reward = 0.
        log_minibatchloss = []
        if augmentation:
            history_blocks = []
            history_actions = []

        obs = env.reset()
        state, block = obs
        if len(state.shape)==2:
            state = state[np.newaxis, :, :]

        pre_action = None
        for t_step in range(env.num_steps):
            count_steps += 1
            ep_len += 1
            action, q_map = get_action(env, FCQ, state, block, epsilon=epsilon, pre_action=pre_action, with_q=True)
            if show_q:
                s0 = deepcopy(state).transpose([1, 2, 0])
                s1 = deepcopy(goal).transpose([1, 2, 0])
                im0 = ax0.imshow(s1)
                s0[action[0], action[1]] = [1, 0, 0]
                # q_map = q_map[0]
                q_map = q_map.transpose([1,2,0]).max(2)
                im = ax1.imshow(s0)
                im2 = ax2.imshow(q_map/q_map.max())
                print('min_q:', q_map.min(), '/ max_q:', q_map.max())
                fig.canvas.draw()

            obs, reward, done = env.step(action[1:])
            next_state, next_block = obs
            if len(next_state.shape)==2:
                next_state = next_state[np.newaxis, :, :]
            episode_reward += reward

            ## save transition to the replay buffer ##
            replay_buffer.add(state, block, action, next_state, next_block, reward, done)

            # trajectory data augmentstion #
            if augmentation and not done:
                traj_samples = sample_trajectories(history_blocks, history_actions, next_state, resolution=resolution)
                for traj_sample in traj_samples:
                    state_re, block_re, action_re = traj_sample
                    replay_buffer.add(state_re, block_re, action_re, next_state, next_block, reward, done)

                history_blocks.append(block)
                history_actions.append(action)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    state = next_state 
                    block = next_block
                    pre_action = action
                    continue
            elif (not learning_starts) and (replay_buffer.size >= learn_start):
                epsilon = start_epsilon
                count_steps = 0
                learning_starts = True
                break

            ## sample from replay buff & update networks ##
            data = [
                    torch.FloatTensor(state).cuda(),
                    torch.FloatTensor(block).cuda(),
                    torch.FloatTensor(next_state).cuda(),
                    torch.FloatTensor(next_block).cuda(),
                    torch.FloatTensor(action).cuda(),
                    torch.FloatTensor([reward]).cuda(),
                    torch.FloatTensor([1 - done]).cuda(),
                    ]
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, _ = calculate_loss(combined_minibatch, FCQ, FCQ_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_minibatchloss.append(loss.data.detach().cpu().numpy())

            if done:
                break
            else:
                state = next_state
                block = next_block
                pre_action = action

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_eplen.append(ep_len)
        log_epsilon.append(epsilon)

        eplog = {
                'Reward': episode_reward,
                'loss': np.mean(log_minibatchloss),
                'EP Len': ep_len,
                'epsilon': epsilon,
                }
        if not wandb_off:
            wandb.log(eplog, count_steps)

        if ne % log_freq == 0:
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
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")

            log_list = [
                    log_returns,  # 0
                    log_loss,  # 1
                    log_eplen,  # 2
                    log_epsilon,  # 3
                    ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_returns[-1] > max_return:
                max_return = log_mean_returns[-1]
                torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                print(" <- Highest Return. Saving the model.")
            else:
                print("")

        if ne % update_freq == 0:
            FCQ_target.load_state_dict(FCQ.state_dict())
            #lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--b1", default=0.1, type=float)
    parser.add_argument("--b2", default=0.25, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--resolution", default=64, type=int)
    ## learning ##
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--buff_size", default=1e4, type=float)
    parser.add_argument("--total_episodes", default=1e5, type=float)
    parser.add_argument("--learn_start", default=3e3, type=float)
    parser.add_argument("--update_freq", default=250, type=int)
    parser.add_argument("--log_freq", default=250, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--small", action="store_true")
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
    render = args.render
    b1 = args.b1
    b2 = args.b2
    max_steps = args.max_steps
    resolution = args.resolution

    # evaluate configuration #
    evaluation = args.evaluate
    model_path = os.path.join("results/models/FCDQN_%s.pth"%args.model_path)
    num_trials = args.num_trials
    show_q = args.show_q

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
    if b1==b2:
        log_name += '_%.2f' %b1
    else:
        log_name += '_%.2f-%.2f' %(b1, b2)
    wandb_off = args.wandb_off
    if not (evaluation or wandb_off):
        wandb.init(project="SKT Palletizing")
        wandb.run.name = log_name
        wandb.config.update(args)
        wandb.run.save()

    env = Floor1(
            resolution=resolution, 
            num_steps=max_steps,
            num_preview=5,
            box_norm=True,
            action_norm=False,
            render=render,
            block_size_min=b1,
            block_size_max=b2
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

    half = args.half
    small = args.small
    continue_learning = args.continue_learning
    if half:
        n_hidden = 8
    else:
        n_hidden = 16
    if small:
        from fcn_resnet import FCQResNetSmall as FCQNet
    else:
        from fcn_resnet import FCQResNet as FCQNet

    if evaluation:
        evaluate(env=env, model_path=model_path, num_trials=num_trials, b1=b1, b2=b2, 
                 show_q=show_q, n_hidden=n_hidden, resolution=resolution)
    else:
        learning(env=env, savename=savename, learning_rate=learning_rate, 
                 batch_size=batch_size, buff_size=buff_size, total_episodes=total_episodes, 
                 learn_start=learn_start, update_freq=update_freq, log_freq=log_freq, 
                 double=double, continue_learning=continue_learning, model_path=model_path, 
                 wandb_off=wandb_off, b1=b1, b2=b2, show_q=show_q, n_hidden=n_hidden,
                 augmentation=augmentation, resolution=resolution)
