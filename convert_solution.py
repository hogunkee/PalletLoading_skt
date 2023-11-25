import pickle
import numpy as np
from matplotlib import pyplot as plt
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
        render=True,
        discrete_block=True,
        max_levels=1,
        show_q=False,
        reward_type=reward_type
    )


for solution in solutions:
    env.reset()
    env.q_value = None

    for bsize, bpose in solution:
        #env.next_block = np.array(bsize) / resolution - 0.01
        h, w = bsize
        x, y = bpose
        env.next_block = (h / resolution - 0.01, w / resolution - 0.01, 0.156)
        center = np.array([y + np.ceil(h/2), x + np.ceil(w/2)])
        action = [0, center[0], center[1]]
        env.step(action)

        # h, w = bsize
        # x, y = bpose
        #
        # center = (x + w/2, y + h/w)
        #
        # action_rot0 = [0, center[0], center[1]]
        # next_block0 = [h - 0.01, w - 0.01]
        # action_rot1 = [1, center[0], center[1]]
        # next_block1 = [w - 0.01, h - 0.01]

exit()

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



new_block = np.random.choice([0.2, 0.3, 0.4, 0.5], 2, True, p=[0.4, 0.3, 0.2, 0.1])
new_block -= 0.01
