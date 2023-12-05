import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--demo_name", default="0", type=str)
parser.add_argument("--resolution", default=10, type=int)
args = parser.parse_args()

resolution = args.resolution
demo_name = 'demo/demos/solution_{}.pkl'.format(args.demo_name)
with open(demo_name, 'rb') as f:
    solutions = pickle.load(f)
    # solutions in [(box_size, box_pose_lefttop)] format

print("* There are {} Demonstrations in {}".format(len(solutions), demo_name))

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
        plt.pause(0.5)
        #plt.show()
