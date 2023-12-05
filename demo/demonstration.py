import os
import time
import pickle
import argparse
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../'))

from utils import get_block_bound, generate_bound_mask
from replay_buffer import OffReplayBuffer as ReplayBuffer
from environment.environment import Floor1 as FloorEnv


class Demonstration(object):
    def __init__(self, resolution=10):
        self.resolution = resolution
        self.set_possible_boxes()

    def set_possible_boxes(self):
        possible_boxes = []
        for h in [2,3,4,5]:
            for w in [2,3,4,5]:
                possible_boxes.append([h, w])
        self.possible_boxes = possible_boxes

    def place(self, block, pose):
        # block: (int, int)
        # pose: left-top position
        resolution = self.resolution
        p_placed = np.zeros([resolution, resolution])
        x, y = pose
        h, w = block
        if y+h > resolution or x+w > resolution:
            return None
        p_placed[y: y+h, x: x+w] = 1
        return p_placed

    def find_possible_placement(self, pallet, block):
        resolution = self.resolution
        for _x in range(resolution):
            for _y in range(resolution):
                _p_block = self.place(block, (_x, _y))
                if _p_block is None:
                    continue
                _p_placed = pallet + _p_block
                if _p_placed.max()==1:
                    return (_x, _y)
        return None

    def divide_pallet(self, num_solutions):
        resolution = self.resolution
        possible_boxes = self.possible_boxes
        pallet = np.zeros([resolution, resolution])

        sol = []
        solutions = []
        count_sol = 0
        p = pallet.copy()
        while count_sol < num_solutions:
            b = possible_boxes[np.random.choice(len(possible_boxes))]
            pose = self.find_possible_placement(p, b)
            if pose is None:
                # initialize
                p = pallet.copy()
                sol = []
            else:
                p_block = self.place(b, pose)
                p = p + p_block
                sol.append((b, pose))

                if p.min()==1 and p.max()==1:
                    # save solution
                    solutions.append(sol)
                    count_sol += 1
                    print('found: %d'%count_sol, end='\r')
                    # initialize
                    p = pallet.copy()
                    sol = []
        return solutions
    
    def find_solutions(self, num_solutions, save_dir='demo/demos/'):
        solutions = self.divide_pallet(num_solutions)
        print(len(solutions), 'solutions found.')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            num_pkl = 0
        else:
            num_pkl = len([pk for pk in os.listdir(save_dir) if pk.startswith('solution') and pk.endswith('.pkl')])
        savename = os.path.join(save_dir, "solution_%d.pkl"%num_pkl)
        with open(savename, "wb") as f:
            pickle.dump(solutions, f)

        print('solutions saved at %s' %savename)
        return solutions, savename

class DemoConverter(object):
    def __init__(self, resolution=10, buff_size=2e5, reward_type='dense'):
        max_levels = 1
        self.resolution = resolution

        self.env = FloorEnv(
                resolution=resolution,
                num_steps=50,
                num_preview=5,
                box_norm=True,
                render=False,
                discrete_block=True,
                max_levels=max_levels,
                show_q=False,
                reward_type=reward_type
            )
        self.replay_buffer = ReplayBuffer([max_levels, resolution, resolution], 2, dim_action=3, max_size=int(buff_size))

    def load_solutions(self, file_path):
        with open(file_path, 'rb') as f:
            solutions = pickle.load(f)
        return solutions

    def convert_solutions(self, solutions):
        resolution = self.resolution
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
                        block = np.array([h / resolution - 0.01, w / resolution - 0.01])
                        by, bx = block
                    elif rot == 1:
                        block = np.array([w / resolution - 0.01, h / resolution - 0.01])
                        bx, by = block
                    bx, by = bx * resolution, by * resolution

                    next_block_bound = get_block_bound(cy, cx, by, bx)
                    min_y, max_y, min_x, max_x = next_block_bound
                    box_placed = np.zeros([resolution, resolution])
                    box_placed[max(min_y, 0): max_y, max(min_x, 0): max_x] = 1

                    previous_state = np.copy(state)
                    state = state + box_placed
                    action = [rot, center[0], center[1]]

                    reward, done = self.env.reward_fuc.get_2d_reward(previous_state, next_block_bound)
                    if state.max()==1 and state.min()==1:
                        done = True

                    traj = [previous_state, block, action, state, reward, done]
                    trajectories.append(deepcopy(traj))

                for i in range(len(trajectories)):
                    state, block, action, next_state, reward, done = trajectories[i]
                    if done:
                        next_block = self.env.make_new_block()[:2]
                    else:
                        next_block = trajectories[i + 1][1]

                    # q_mask = np.ones((2, resolution, resolution))
                    # next_q_mask = np.ones((2, resolution, resolution))
                    q_mask = generate_bound_mask(np.expand_dims(state, 0), block)
                    next_q_mask = generate_bound_mask(np.expand_dims(next_state, 0), next_block)

                    self.replay_buffer.add(state, block, q_mask, action, next_state, next_block, next_q_mask, reward, done)
                    print('size of replay: %d' %self.replay_buffer.size, end='\r')
        print('conversion done.')

    def save_replay(self, save_dir='demo/replay/'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            num_pkl = 0
        else:
            num_pkl = len([pk for pk in os.listdir(save_dir) if pk.startswith('replay_') and pk.endswith('.pkl')])
        savename = os.path.join(save_dir, "replay_%d_B.pkl"%num_pkl)
        with open(savename, "wb") as f:
            pickle.dump(self.replay_buffer, f)
        print('replay data saved at %s' %savename)


if __name__=='__main__':
    # solution should be in [(box_size, box_pose), ...] format.
    # -> (box_size, box_pose_lefttop)

    # pallet should be in grid map format.

    # possible boxes e.g.,
    # [[2,2], [3,3], [5,5]]
    # [[1,1], [2,1], [1,2], [2,2]]
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="find_demos", type=str, help=["find_demos","convert_demos"])
    parser.add_argument("--demo_name", default="", type=str)
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--num_solutions", default=10, type=int)
    args = parser.parse_args()

    if args.work == "find_demos":
        st = time.time()
        Demo = Demonstration(resolution=args.resolution)
        solutions, demo_name = Demo.find_solutions(args.num_solutions)
        et = time.time()
        print(et - st, 'seconds.')

        Converter = DemoConverter(resolution=args.resolution, buff_size=2e5, reward_type='dense')
        solutions = Converter.load_solutions(demo_name)
        Converter.convert_solutions(solutions)
        Converter.save_replay()

    elif args.work == "convert_demos":
        demo_name = "demo/demos/solution_{}.pkl".format(args.demo_name)
        Converter = DemoConverter(resolution=args.resolution, buff_size=2e5, reward_type='dense')
        solutions = Converter.load_solutions(demo_name)
        Converter.convert_solutions(solutions)
        Converter.save_replay()
