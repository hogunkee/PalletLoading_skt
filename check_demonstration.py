import pickle
import numpy as np
from matplotlib import pyplot as plt

filename = 'results/demo/solution_3.pkl'
with open(filename, 'rb') as f:
    solutions = pickle.load(f)
    # solutions in [(box_size, box_pose_lefttop)] format

print(len(solutions))

resolution = 10

for idx in np.random.choice(len(solutions), 10):
    pallet = np.zeros([resolution, resolution])
    solution = solutions[idx]
    for bsize, bpose in solution:
        h, w = bsize
        x, y = bpose
        if not ((y+h)<=resolution and (x+w)<=resolution):
            print('Out of range!')
            print('solution:', idx)
            exit()
        pallet[y: y+h, x: x+w] += 1
        if pallet.max()>1:
            print('Collision!')
            print('solution:', idx)
            exit()
    plt.imshow(pallet, vmin=0.0, vmax=1.0)
    plt.show()
