import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class Floor1(object):
    def __init__(self, 
                 resolution=512,
                 num_steps=100,
                 num_preview=5,
                 box_norm=False,
                 action_norm=False,
                 render=False,
                 block_size_min=0.2,
                 block_size_max=0.4,
                 show_q=False
                 ):
        """
        resolution: int (default: 512)
            image resolution of the palette.

        num_steps: int (default: 100)
            number of timesteps until scenario ends.

        num_preview: int (default: 5)
            number of next blocks to preview.

        box_norm: bool (default: False)
            if True, the width and height of the box are given between 0 and 1.

        action_norm: bool (default: False)
            if True, the placement position should be given between 0 and 1.

        render: bool (default: False)
            rendering the current palette state.
        """
        self.resolution = resolution
        self.num_steps = num_steps
        self.num_preview = num_preview
        self.box_norm = box_norm
        self.action_norm = action_norm
        self.render = render
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max
        self.show_q = show_q
        self.q_value = None

        if self.render:
            if self.show_q:
                plot0 = plt.subplot2grid((2,5), (0,0))
                plot1 = plt.subplot2grid((2,5), (0,1))
                plot2 = plt.subplot2grid((2,5), (0,2))
                plot3 = plt.subplot2grid((2,5), (0,3))
                plot4 = plt.subplot2grid((2,5), (0,4))
                plot5 = plt.subplot2grid((2,5), (1,0), colspan=5)
                plot0.set_title('Previous state')
                plot1.set_title('Current state')
                plot2.set_title('Next block')
                plot3.set_title('Q-value 0')
                plot4.set_title('Q-value 1')
                plot5.set_title('Next blocks')
                self.plots = [plot0, plot1, plot2, plot3, plot4, plot5]
            else:
                plot0 = plt.subplot2grid((2,3), (0,0))
                plot1 = plt.subplot2grid((2,3), (0,1))
                plot2 = plt.subplot2grid((2,3), (0,2))
                plot3 = plt.subplot2grid((2,3), (1,0), colspan=3)
                plot0.set_title('Previous state')
                plot1.set_title('Current state')
                plot2.set_title('Next block')
                plot3.set_title('Next blocks')
                self.plots = [plot0, plot1, plot2, plot3]
            for i, p in enumerate(self.plots):
                p.set_xticks([])
                p.set_yticks([])
                if i==0 or i==1:
                    p.axis('off')
            plt.show(block=False)

    def reset(self):
        self.state, self.next_block = self.init_scenario()
        if self.render:
            self.render_current_state(self.state)
        
        next_block = np.array(self.next_block)
        obs = (self.state, next_block)
        return obs
        # return self.state, self.next_block

    def render_current_state(self, previous_state=None, box=None):
        pad = int(0.1 * self.resolution)
        state_pad = np.ones([int(1.2*self.resolution), int(1.2*self.resolution), 3]) * 0.7
        pad_mask = np.ones(state_pad.shape[:2])
        pad_mask[pad:-pad, pad:-pad] = 0

        pre_state_pad = state_pad.copy()
        y, x = np.where(previous_state==0)
        pre_state_pad[y + pad, x + pad] = (1, 1, 1)
        y, x = np.where(previous_state==1)
        pre_state_pad[y + pad, x + pad] = (0, 0, 0)
        y, x = np.where(previous_state==2)
        pre_state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(np.all(pre_state_pad!=[0.7, 0.7, 0.7], axis=-1) & (pad_mask==1))
        pre_state_pad[y, x] = (1, 0, 0)
        self.plots[0].imshow(pre_state_pad)

        if self.show_q:
            if self.q_value is not None:
                if len(self.q_value.shape)==2:
                    q_value_pad = np.pad(self.q_value, (pad, pad), 'constant', \
                                        constant_values=self.q_value.min())
                    self.plots[3].imshow(q_value_pad)
                elif len(self.q_value.shape)==3:
                    i, _q in enumerate(self.q_value):
                        q_value_pad = np.pad(_q, (pad, pad), 'constant', \
                                            constant_values=_q.min())
                        self.plots[3 + i].imshow(q_value_pad)
            else:
                q_value_empty = np.zeros_like(pad_mask)
                self.plots[3].imshow(q_value_empty)

        state_pad = np.ones([int(1.2*self.resolution), int(1.2*self.resolution), 3]) * 0.7
        y, x = np.where(self.state==0)
        state_pad[y + pad, x + pad] = (1, 1, 1)
        y, x = np.where(self.state==1)
        state_pad[y + pad, x + pad] = (0, 0, 0)
        if box is not None:
            min_y, max_y, min_x, max_x = np.array(box) + pad
            state_pad[min_y: max_y, min_x: max_x] = (0, 0, 1)
        y, x = np.where(self.state==2)

        state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(np.all(state_pad!=[0.7, 0.7, 0.7], axis=-1) & (pad_mask==1))
        state_pad[y, x] = (1, 0, 0)
        self.plots[1].imshow(state_pad)

        block_figures = None
        if self.step_count==0:
            next_blocks = [self.next_block] + self.block_que[:-1]
        else:
            next_blocks = self.block_que
        #for i, b in enumerate([self.next_block] + self.block_que[:-1]):
        for i, b in enumerate(next_blocks):
            cy, cx = int(0.6*self.resolution), int(0.6*self.resolution)
            b = np.round(np.array(b) * self.resolution).astype(int)
            by, bx = b
            min_y = np.round(cy - (by-1e-5)/2).astype(int)
            min_x = np.round(cx - (bx-1e-5)/2).astype(int)
            max_y = np.round(cy + (by-1e-5)/2).astype(int)
            max_x = np.round(cx + (bx-1e-5)/2).astype(int)

            block_fig = np.ones_like(state_pad)
            block_fig[min_y: max_y, min_x: max_x] = [0, 0, 0]

            if i==0:
                self.plots[2].imshow(block_fig)
            else:
                if block_figures is None:
                    block_figures = block_fig
                else:
                    block_figures = np.concatenate([block_figures, block_fig], axis=1)
        self.plots[-1].imshow(block_figures)
        plt.draw()
        # plt.pause(0.01)
        plt.pause(1)

    def get_next_block(self):
        next_block = self.block_que.pop(0)
        new_block = np.random.uniform(self.block_size_min, self.block_size_max, 2)
        self.block_que.append(new_block)
        return next_block

    def init_scenario(self):
        state = np.zeros([self.resolution, self.resolution])
        #state = np.zeros([self.resolution-2, self.resolution-2])
        #state = np.pad(state, (1, 1), 'constant', constant_values=(1))
        self.step_count = 0

        # next block: (height, width)
        self.block_que = np.random.uniform(self.block_size_min, self.block_size_max, [self.num_preview, 2]).tolist()

        next_block = self.get_next_block()
        return state, next_block

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        out_of_range = False
        collision = False
        episode_end = False

        # previous state #
        previous_state = self.state

        # denormalize action #
        if self.action_norm:
            action_pos = np.array(action[1:]) * self.resolution
        else:
            action_pos = np.array(action[1:])
        action_rot = action[0]

        # make box region #
        cy, cx = action_pos
        if self.box_norm:
            next_block = np.array(self.next_block) * self.resolution
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        if action_rot==0:
            by, bx = next_block
        elif action_rot==1:
            bx, by = next_block
            
        min_y = np.round(cy - (by-1e-5)/2).astype(int)
        min_x = np.round(cx - (bx-1e-5)/2).astype(int)
        max_y = np.round(cy + (by-1e-5)/2).astype(int)
        max_x = np.round(cx + (bx-1e-5)/2).astype(int)

        # check out of range #
        if min_y < 0 or min_x < 0:
            out_of_range = True
        if max_y > self.resolution or max_x > self.resolution:
            out_of_range = True
        if out_of_range:
            reward = 0.0
            episode_end = True

        # check collision #
        box_placed = np.zeros([self.resolution, self.resolution])
        box_placed[min_y: max_y, min_x: max_x] = 1
        self.state = self.state + box_placed
        if len(np.where(self.state>1)[0]) > 0:
            collision = True
            reward = 0.0
            episode_end = True

        # if no OOR of collision, the placement succeeds #
        if not (out_of_range or collision):
            reward = 1.0
            episode_end = False

        if self.render:
            self.render_current_state(previous_state, [min_y, max_y, min_x, max_x])

        self.next_block = self.get_next_block()
        if self.box_norm:
            next_block = self.next_block
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        
        next_block = np.array(next_block)
        obs = (self.state, next_block)
        # return state, next_block, reward, episode_end
        return obs, reward, episode_end

if __name__=='__main__':
    box_norm = True
    action_norm = True
    env = Floor1(resolution=32, box_norm=box_norm, action_norm=action_norm, render=False, block_size_min=0.1, block_size_max=0.25)
    # state, next_block = env.reset()

    total_reward = 0.
    num_episodes = 100 
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.
        # print(f'Episode {ep} starts.')
        for i in range(100):
            if action_norm:
                random_action = np.random.uniform(0.1, 0.9, 2)
                action = random_action.tolist()
            else:
                random_action = np.random.uniform(0.1, 0.9, 2) * env.resolution
                action = np.round(random_action).astype(int).tolist()
            # print('action:', action)
            # state, next_block, reward, end = env.step(action)
            obs, reward, end = env.step(action)
            ep_reward += reward
            if end:
                # print('Episode ends.')
                break
        # print("    ep_reward: ", ep_reward)
        total_reward += ep_reward
    avg_score = total_reward / num_episodes
    print("average score: ", avg_score)
