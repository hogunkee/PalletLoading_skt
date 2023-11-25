import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

def place(block, pose):
    # block: (int, int)
    # pose: left-top position
    p_placed = np.zeros([resolution, resolution])
    x, y = pose
    h, w = block
    if y+h > resolution or x+w > resolution:
        return None
    p_placed[y: y+h, x: x+w] = 1
    return p_placed

def find_possible_placement(pallet, block):
    for _x in range(resolution):
        for _y in range(resolution):
            _p_block = place(block, (_x, _y))
            if _p_block is None:
                continue
            _p_placed = pallet + _p_block
            if _p_placed.max()==1:
                return (_x, _y)
    return None

def divide_pallet(num_solutions, resolution):
    pallet = np.zeros([resolution, resolution])

    sol = []
    solutions = []
    count_sol = 0
    p = pallet.copy()
    while count_sol < num_solutions:
        b = possible_boxes[np.random.choice(len(possible_boxes))]
        pose = find_possible_placement(p, b)
        if pose is None:
            # initialize
            p = pallet.copy()
            sol = []
        else:
            p_block = place(b, pose)
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

def pose_center2lefttop(pose, block):
    return

def pose_lefttop2center(pose, block):
    return

if __name__=='__main__':
    # solution should be in [(box_size, box_pose), ...] format.
    # -> (box_size, box_pose_lefttop)

    # pallet should be in grid map format.

    # [[2,2], [3,3], [5,5]]
    # [[1,1], [2,1], [1,2], [2,2]]

    num_solutions = 10000
    resolution = 10
    # possible_boxes = [[2,2], [3,3], [5,5]]
    possible_boxes = []
    for h in [2,3,4,5]:
        for w in [2,3,4,5]:
            possible_boxes.append([h, w])
    print(possible_boxes)

    st = time.time()
    solutions = divide_pallet(num_solutions, resolution)
    print(len(solutions), 'solutions found.')
    et = time.time()
    print(et - st, 'seconds.')

    #from pprint import pprint
    #pprint(global_solutions)

    num_pkl = len([pk for pk in os.listdir('results/demo/') if pk.startswith('solution') and pk.endswith('.pkl')])
    savename = "results/demo/solution_%d.pkl" %num_pkl
    with open(savename, "wb") as f:
        pickle.dump(solutions, f)
    print('solutions saved at %s' %savename)
