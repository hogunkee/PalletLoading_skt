import time
import pickle
import numpy as np
from matplotlib import pyplot as plt


global count_sol

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

def fit(pallet, block):
    positions = find_possible_placement(pallet, block)
    for pose in positions:
        p_block = place(block, pose)
        if p_block is None:
            return False, None
        p_placed = pallet + p_block
        if p_placed.min()==1 and p_placed.max()==1:
            return True, pose
    return False, None

def find_possible_placement(pallet, block):
    for _x in range(resolution):
        for _y in range(resolution):
            _p_block = place(block, (_x, _y))
            if _p_block is None:
                continue
            _p_placed = pallet + _p_block
            if _p_placed.max()==1:
                return [(_x, _y)]
    return []

def merge(solution1, solution2):
    solutions = []
    for s1 in solution1:
        for s2 in solution2:
            s = s1 + s2
            solutions.append(s)
    return solutions

def divide_pallet(p):
    global count_sol
    solutions = []
    for b in possible_boxes:
        is_fit, pose = fit(p, b)
        if is_fit:
            sol = [(b, pose)]
            solutions.append(sol)
            count_sol += 1
            print('found: %d'%count_sol, end='\r')
        else:
            positions = find_possible_placement(p, b)
            for pose in positions:
                p_placed = place(b, pose)
                if p_placed is None:
                    continue
                p_left = p + p_placed
                assert p_left.min()>=0 and p_left.max()<=1
                sol_box = [[(b, pose)]]
                #sol_box = divide_pallet(1 - p_placed)
                sol_left = divide_pallet(p_left)
                solutions = solutions + merge(sol_box, sol_left)
    return solutions

def pose_center2lefttop(pose, block):
    return

def pose_lefttop2center(pose, block):
    return

if __name__=='__main__':
    # solution should be in [(box_size, box_pose), ...] format.
    # -> (box_size, box_pose_lefttop, box_pose_center)

    # pallet should be in grid map format.

    # [[2,2], [3,3], [5,5]]
    # [[1,1], [2,1], [1,2], [2,2]]

    resolution = 10
    # possible_boxes = [[2,2], [3,3], [5,5]]
    possible_boxes = []
    for h in [2,3,4,5]:
        for w in [2,3,4,5]:
            possible_boxes.append([h, w])
    print(possible_boxes)
    count_sol = 0

    st = time.time()
    pallet = np.zeros([resolution, resolution])
    solutions = divide_pallet(pallet)
    et = time.time()

    print(et - st, 'seconds.')
    print(len(solutions), 'solutions found.')
    from pprint import pprint
    pprint(solutions)

    savename = "solution_all.pkl"
    with open(savename, "wb") as f:
        pickle.dump(solutions, f)
    print('solutions saved at %s' %savename)
