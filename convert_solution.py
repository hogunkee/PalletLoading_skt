import pickle
import numpy as np
from matplotlib import pyplot as plt
from utils import get_block_bound
from environment.environment import Floor1 as FloorEnv

filename = 'solution_3.pkl'
with open(filename, 'rb') as f:
    solutions = pickle.load(f)
    # solutions in [(box_size, box_pose_lefttop)] format

print(len(solutions))

resolution = 10
reward_type = 'dense'
env = FloorEnv(
        resolution=resolution,
        num_steps=50,
        num_preview=5,
        box_norm=True,
        render=False,
        discrete_block=True,
        max_levels=1,
        show_q=False,
        reward_type=reward_type
    )


for solution in solutions:
    for rot in range(2):
        state = np.zeros([resolution, resolution])
        trajectories = []
        for bsize, bpose in solution:
            h, w = bsize
            x, y = bpose
            center = np.array([y + np.ceil(h/2), x + np.ceil(w/2)])
            action = [rot, center[0], center[1]]

            action_rot = action[0]
            cy, cx = np.array(action[1:])
            if action_rot == 0:
                block = np.array([h / resolution - 0.01, w / resolution - 0.01, 0.156]) * resolution
                by, bx, _ = block
            elif action_rot == 1:
                block = np.array([w / resolution - 0.01, h / resolution - 0.01, 0.156]) * resolution
                bx, by, _ = block

            next_block_bound = get_block_bound(cy, cx, by, bx)
            min_y, max_y, min_x, max_x = next_block_bound
            box_placed = np.zeros([resolution, resolution])
            box_placed[max(min_y, 0): max_y, max(min_x, 0): max_x] = 1
            previous_state = np.copy(state)
            state = state + box_placed

            reward, done = env.reward_fuc.get_2d_reward(previous_state, next_block_bound)

            traj = [previous_state, block, action, state, reward, done]
            trajectories.append(traj)

    # get q_mask
    next_q_mask = np.ones((2, resolution, resolution))

    [state, block, action, next_state, next_q_mask, reward, done] # next_block, next_q_mask needed.
    traj = [state, block, action, next_state, next_block, next_q_mask, reward, done]



# old version #
for solution in []:
    env.reset()
    env.q_value = None
    state = np.zeros([resolution, resolution])
    for bsize, bpose in solution:
        h, w = bsize
        x, y = bpose
        env.next_block = (h / resolution - 0.01, w / resolution - 0.01, 0.156)
        center = np.array([y + np.ceil(h/2), x + np.ceil(w/2)])
        action = [0, center[0], center[1]]
        obs, reward, done = env.step(action)
        print('reward:', reward)

        action_rot = action[0]
        cy, cx = np.array(action[1:])
        if action_rot == 0:
            next_block = np.array([h / resolution - 0.01, w / resolution - 0.01, 0.156]) * resolution
            by, bx, _ = next_block
        elif action_rot == 1:
            next_block = np.array([w / resolution - 0.01, h / resolution - 0.01, 0.156]) * resolution
            bx, by, _ = next_block

        next_block_bound = get_block_bound(cy, cx, by, bx)
        min_y, max_y, min_x, max_x = next_block_bound
        box_placed = np.zeros([resolution, resolution])
        box_placed[max(min_y, 0): max_y, max(min_x, 0): max_x] = 1
        previous_state = np.copy(state)
        state = state + box_placed

        reward, episode_end = env.reward_fuc.get_2d_reward(previous_state, next_block_bound)
        next_q_mask = np.ones((2, resolution, resolution))

        traj = [state, block, action, next_state, next_block, next_q_mask, reward, done]
        print('reward calculated:', reward)
        print()

    env.reset()
    env.q_value = None
    state = np.zeros([resolution, resolution])
    for bsize, bpose in solution:
        # flip the block
        h, w = bsize
        x, y = bpose
        env.next_block = (w / resolution - 0.01, h / resolution - 0.01, 0.156)
        center = np.array([y + np.ceil(h/2), x + np.ceil(w/2)])
        action = [1, center[0], center[1]]
        obs, reward, done = env.step(action)
        print('reward:', reward)

        action_rot = action[0]
        cy, cx = np.array(action[1:])
        if action_rot==0:
            next_block = np.array([h / resolution - 0.01, w / resolution - 0.01, 0.156]) * resolution
            by, bx, _ = next_block
        elif action_rot==1:
            next_block = np.array([w / resolution - 0.01, h / resolution - 0.01, 0.156]) * resolution
            bx, by, _ = next_block

        next_block_bound = get_block_bound(cy, cx, by, bx)
        min_y, max_y, min_x, max_x = next_block_bound
        box_placed = np.zeros([resolution, resolution])
        box_placed[max(min_y, 0): max_y, max(min_x, 0): max_x] = 1
        previous_state = np.copy(state)
        state = state + box_placed

        reward, episode_end = env.reward_fuc.get_2d_reward(previous_state, next_block_bound)
        print('reward calculated:', reward)
        print()