import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from utils import get_block_bound
from replay_buffer import ReplayBuffer
from environment.environment import Floor1 as FloorEnv

filename = 'solution_3.pkl'
with open(filename, 'rb') as f:
    solutions = pickle.load(f)
    # solutions in [(box_size, box_pose_lefttop)] format

print(len(solutions))

max_levels = 1
resolution = 10
buff_size = 2e5
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
replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

for solution in solutions:
    for rot in range(2):
        state = np.zeros([resolution, resolution])
        trajectories = []
        for bsize, bpose in solution:
            h, w = bsize
            x, y = bpose
            center = np.array([y + np.ceil(h/2), x + np.ceil(w/2)])

            cy, cx = center #np.array(action[1:])
            if rot == 0:
                block = np.array([h / resolution - 0.01, w / resolution - 0.01]) * resolution
                by, bx = block
            elif rot == 1:
                block = np.array([w / resolution - 0.01, h / resolution - 0.01]) * resolution
                bx, by = block

            next_block_bound = get_block_bound(cy, cx, by, bx)
            min_y, max_y, min_x, max_x = next_block_bound
            box_placed = np.zeros([resolution, resolution])
            box_placed[max(min_y, 0): max_y, max(min_x, 0): max_x] = 1

            previous_state = np.copy(state)
            state = state + box_placed
            action = [rot, center[0], center[1]]

            reward, done = env.reward_fuc.get_2d_reward(previous_state, next_block_bound)
            if state.max()==1 and state.min()==1:
                done = True

            traj = [previous_state, block, action, state, reward, done]
            trajectories.append(deepcopy(traj))

        for i in range(len(trajectories)):
            state, block, action, next_state, reward, done = trajectories[i]
            if done:
                next_block = env.make_new_block()[:2]
            else:
                next_block = trajectories[i + 1][1]
            next_q_mask = np.ones((2, resolution, resolution))

            replay_buffer.add(state, block, action, next_state, next_block, next_q_mask, reward, done)
            print('size of replay: %d' %replay_buffer.size, end='\r')

print()
num_pkl = len([pk for pk in os.listdir() if pk.startswith('replay_') and pk.endswith('.pkl')])
savename = "replay_%d.pkl" %num_pkl
with open(savename, "wb") as f:
    pickle.dump(replay_buffer, f)
print('replay data saved at %s' %savename)
