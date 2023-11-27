import torch
import datetime
import time
import os
import wandb

from utils import *
from replay_buffer import ReplayBuffer, OnPolicyMemory
from agent.DQN import DQN_Agent
from agent.D_PPO import D_PPO_Agent
from agent.D_SAC import D_SAC_Agent
from agent.D_TAC import D_TAC_Agent

def evaluate(
        env,
        agent=None,
        algo='',
        model_path='',
        num_trials=10,
        show_q=False,
        resolution=20,
        max_levels=1,
        use_bound_mask=False,
        use_floor_mask=False,
        use_projection=False,
        use_coordnconv=False,
        print_info=False,
    ):
    if agent is None:
        if algo == "DQN":
            agent = DQN_Agent(max_levels, resolution, train=False,
                        model_path=model_path, use_coordnconv=use_coordnconv)
        elif algo == "D-PPO":
            agent = D_PPO_Agent(max_levels, resolution, train=False,
                        model_path=model_path, use_coordnconv=use_coordnconv)
        elif algo == "D-SAC":
            agent = D_SAC_Agent(max_levels, resolution, train=False,
                        model_path=model_path, use_coordnconv=use_coordnconv)
        elif algo == "D-TAC":
            agent = D_TAC_Agent(max_levels, resolution, train=False,
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

            action, q_map, *_ = agent.get_action(state, block,
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
            print("Current: R{:.2f}, B{:.2f}, P{:.3f}".format(log_returns[-1],log_eplen[-1],log_pf[-1]), end=" / ")
            print("Mean: R{:.2f}, B{:.2f}, P{:.3f}".format(np.mean(log_returns),np.mean(log_eplen),np.mean(log_pf)))

    if print_info:
        print()
        print("="*80)
        print("Evaluation Done.")
        print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
        print("Mean episode length: {}".format(np.mean(log_eplen)))

    return np.mean(log_eplen), np.mean(log_pf)

    
def DQNlearning(
        env,
        savename,
        log_dir='',
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
    ):
    agent = DQN_Agent(max_levels, resolution, True,
                  learning_rate, model_path, double, use_coordnconv)

    replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    log_returns, log_loss, log_eplen, log_test_len, log_test_pf, log_step = [], [], [], [], [], []

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
            replay_buffer.add(state, block, action, next_state, next_block, next_q_mask, reward, done)

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
            np.save(log_dir + '%s' %savename, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                torch.save(agent.FCQ.state_dict(), model_path + '/%s.pth' % savename)
                print(" <- Highest Return. Saving the model.")
            else:
                torch.save(agent.FCQ.state_dict(), model_path + '/%s_last.pth' % savename)
                print("")

            agent.train_on_off(train=True)

    print('Training finished.')

    
def DPPOlearning(
        env,
        savename,
        log_dir='',
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
        gamma=0.99,
        train_interval = 50
    ):
    agent = D_PPO_Agent(max_levels, resolution, True,
                  learning_rate, model_path, use_coordnconv, gamma=gamma)

    replay_buffer = OnPolicyMemory([max_levels, resolution, resolution], 3, gamma=gamma, lam=0.95, lim=train_interval)

    log_returns, log_loss, log_eplen, log_test_len, log_test_pf, log_step = [], [], [], [], [], []

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

        for t in range(env.num_steps):
            count_steps += 1
            ep_len += 1

            if use_bound_mask:
                q_mask = generate_bound_mask(state, block)

            if use_floor_mask:
                q_mask = generate_floor_mask(state, block, q_mask)

            action, probs, value = agent.get_action(state, block,
                                             with_q=True, deterministic=False,
                                             q_mask=q_mask, p_project=p_projection)
            
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
            replay_buffer.append(state, block, action, reward, value, probs, next_q_mask)

            # termination of env, or by truncation due to memory size     
            if (t == env.num_steps - 1 or replay_buffer._size == train_interval):
                next_state = torch.FloatTensor(next_state).unsqueeze(0).cuda()
                next_block = torch.FloatTensor(next_block).unsqueeze(0).cuda()
                v_last = agent.V(next_state, next_block).squeeze()
                replay_buffer.compute_values(v_last)
                break
            elif done:
                v_last = torch.zeros([2, resolution, resolution]).cuda()
                replay_buffer.compute_values(v_last)
                break
            
            state, block = next_state, next_block

        if replay_buffer._size == train_interval:
            minibatch = replay_buffer.load()
            loss = agent.learn(minibatch)
            log_minibatchloss.append(loss.detach().cpu().numpy())
            replay_buffer.reset()

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
            np.save(log_dir + '%s' %savename, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                torch.save(agent.V.state_dict(), model_path + '/%s_critic.pth' % savename)
                torch.save(agent.pi.state_dict(), model_path + '/%s_actor.pth' % savename)
                print(" <- Highest Return. Saving the model.")
            else:
                torch.save(agent.V.state_dict(), model_path + '/%s_critic_last.pth' % savename)
                torch.save(agent.pi.state_dict(), model_path + '/%s_actor_last.pth' % savename)
                print("")

            agent.train_on_off(train=True)

    print('Training finished.')

    
def DSAClearning(
        env,
        savename,
        log_dir='',
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
        train_interval=100
    ):
    agent = D_SAC_Agent(max_levels, resolution, True,
                  learning_rate, model_path, double, use_coordnconv)

    replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    log_returns, log_loss, log_eplen, log_test_len, log_test_pf, log_step = [], [], [], [], [], []

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
            replay_buffer.add(state, block, action, next_state, next_block, next_q_mask, reward, done)

            state, block = next_state, next_block

            if replay_buffer.size < learn_start:
                if done: break
                else: continue

            if done: break

        if count_steps % train_interval == 0:
            for _ in range(5):
                minibatch = replay_buffer.sample(batch_size)
                loss = agent.learn(minibatch, tau=tau)
                log_minibatchloss.append(loss)

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
            np.save(log_dir + '%s' %savename, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                torch.save(agent.Q.state_dict(), model_path + '%s_critic.pth' % (savename))
                torch.save(agent.pi.state_dict(), model_path + '%s_actor.pth' % (savename))
                print(" <- Highest Return. Saving the model.")
            else:
                torch.save(agent.Q.state_dict(), model_path + '%s_critic_last.pth' % (savename))
                torch.save(agent.pi.state_dict(), model_path + '%s_actor_last.pth' % (savename))
                print("")

            agent.train_on_off(train=True)

    print('Training finished.')

def DTAClearning(
        env,
        savename,
        log_dir='',
        learning_rate=3e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_episodes=1e6,
        learn_start=1e4,
        log_freq=1e3,
        tau=1e-3,
        q_prime=1.2, # tsallis hyperparameter
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
        train_interval=100
    ):
    agent = D_TAC_Agent(max_levels, resolution, True,
                  learning_rate, model_path, double, use_coordnconv, q_prime=q_prime)

    replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    log_returns, log_loss, log_eplen, log_test_len, log_test_pf, log_step = [], [], [], [], [], []

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
            replay_buffer.add(state, block, action, next_state, next_block, next_q_mask, reward, done)

            state, block = next_state, next_block

            if replay_buffer.size < learn_start:
                if done: break
                else: continue

            if done: break

        if count_steps % train_interval == 0:
            for _ in range(5):
                minibatch = replay_buffer.sample(batch_size)
                loss = agent.learn(minibatch, tau=tau)
                log_minibatchloss.append(loss)

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
            np.save(log_dir + '%s' %savename, numpy_log)

            if test_pf > max_return:
                max_return = test_pf
                torch.save(agent.Q.state_dict(), model_path + '%s_critic.pth' % (savename))
                torch.save(agent.pi.state_dict(), model_path + '%s_actor.pth' % (savename))
                print(" <- Highest Return. Saving the model.")
            else:
                torch.save(agent.Q.state_dict(), model_path + '%s_critic_last.pth' % (savename))
                torch.save(agent.pi.state_dict(), model_path + '%s_actor_last.pth' % (savename))
                print("")

            agent.train_on_off(train=True)

    print('Training finished.')

