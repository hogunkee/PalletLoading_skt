import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



class Renderer():
    def __init__(self, resolution, show_q=False):
        if show_q:
            plt.figure(figsize=(10,5))
            plot0 = plt.subplot2grid((2,4), (0,0))
            plot1 = plt.subplot2grid((2,4), (0,1))
            plot2 = plt.subplot2grid((2,4), (0,2))
            plot3 = plt.subplot2grid((2,4), (0,3))
            plot4 = plt.subplot2grid((2,4), (1,3))
            plot5 = plt.subplot2grid((2,4), (1,0), colspan=3)
            plot0.set_title('Previous state')
            plot1.set_title('Current state')
            plot2.set_title('Next block')
            plot3.set_title('Q-value rot-0')
            plot4.set_title('Q-value rot-90')
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
            if i==0 or i==1: p.axis('off')
        plt.show(block=False)

        self.resolution = resolution
        self.show_q = show_q

    def render_current_state(self, state, next_blocks, previous_state, q_value=None, box=None):
        if len(np.shape(state)) == 3:
            state = state[0]
            previous_state = previous_state[0]
        
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
            if q_value is not None:
                if len(q_value.shape)==2:
                    q_value_pad = np.pad(q_value, (pad, pad), 'constant', \
                                        constant_values=q_value.min())
                    self.plots[3].imshow(q_value_pad)
                elif len(q_value.shape)==3:
                    for i, _q in enumerate(q_value):
                        q_value_pad = np.pad(_q, (pad, pad), 'constant', \
                                            constant_values=_q.min())
                        self.plots[3 + i].imshow(q_value_pad)
            else:
                q_value_empty = np.zeros_like(pad_mask)
                self.plots[3].imshow(q_value_empty)
                self.plots[4].imshow(q_value_empty)

        state_pad = np.ones([int(1.2*self.resolution), int(1.2*self.resolution), 3]) * 0.7
        y, x = np.where(state==0)
        state_pad[y + pad, x + pad] = (1, 1, 1)
        y, x = np.where(state==1)
        state_pad[y + pad, x + pad] = (0, 0, 0)
        if box is not None:
            min_y, max_y, min_x, max_x = np.array(box) + pad
            state_pad[min_y: max_y, min_x: max_x] = (0, 0, 1)
        y, x = np.where(state==2)

        state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(np.all(state_pad!=[0.7, 0.7, 0.7], axis=-1) & (pad_mask==1))
        state_pad[y, x] = (1, 0, 0)
        self.plots[1].imshow(state_pad)

        block_figures = None
            
        for i, b in enumerate(next_blocks):
            cy, cx = int(0.6*self.resolution), int(0.6*self.resolution)
            b = np.round(np.array(b) * self.resolution).astype(int)
            if len(b) > 2: b = b[:2]
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


class RewardFunc():
    def __init__(self, reward_type, stability_sim=None, max_levels=1, sim_render=False):
        self.reward_type = reward_type
        self.stability_sim = stability_sim
        self.max_levels = max_levels
        self.sim_render = sim_render

    def get_pad_from_scene(self, state, pad_boundary=True):
        state = state.astype(bool).astype(float).copy()

        if pad_boundary:
            state = np.pad(state, (1,1), 'constant', constant_values=(1))
            
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(state, kernel).astype(bool).astype(float)

        if pad_boundary:
            state = state[1:-1, 1:-1]
            dilated = dilated[1:-1, 1:-1]

        return dilated - state
    
    def get_2d_reward(self, state, block_bound):
        min_y, max_y, min_x, max_x = block_bound

        # check out of range #
        out_of_range = False
        if min_y < 0 or min_x < 0:
            out_of_range = True
        elif max_y > np.shape(state)[0] or max_x > np.shape(state)[1]:
            out_of_range = True

        # check collision #
        box_placed = np.zeros(np.shape(state))    
        box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1 
        next_state = state + box_placed

        collision = False
        if len(np.where(next_state>self.max_levels)[0]) > 0:
           collision = True            
        
        reward = 0.0
        episode_end = False

        if out_of_range or collision:
            reward = 0.0
            episode_end = True

        elif self.reward_type=='binary':
            reward = 1.0

        elif self.reward_type=='dense':
            C = 1/100
            p_box = self.get_pad_from_scene(box_placed, False).sum()
            p_current = self.get_pad_from_scene(state).sum()
            p_next = self.get_pad_from_scene(next_state).sum()
            reward = C * (p_box + p_current - p_next)

            # p_current = self.get_pad_from_scene(state)
            # reward = 1.0 + 100.0*np.multiply(p_current,box_placed).sum()/np.ones(np.shape(state)).sum()

        elif self.reward_type=='dense2':
            C = 1/100
            p_box = self.get_pad_from_scene(box_placed, False).sum()
            p_current = self.get_pad_from_scene(state).sum()
            p_next = self.get_pad_from_scene(next_state).sum()
            reward = C * (p_box + p_current - p_next) + 0.2

        return reward, episode_end

    def get_3d_reward(self, state, block_bound, stacked_history, level_map, box_level):
        if np.max(level_map) > self.max_levels:
            reward, episode_end = 0.0, True # 0.0, True
            return reward, episode_end

        pose_list = stacked_history["pose_list"]
        quat_list = stacked_history["quat_list"]
        scale_list = stacked_history["scale_list"]

        stability_ = self.stability_sim.stacking(pose_list, quat_list, scale_list,
                                                 render=self.sim_render)
        
        if not stability_:
            reward, episode_end = 0.0, True # 0.0, True
        else:
            reward, episode_end = self.get_2d_reward(state[box_level-1], block_bound)

        return reward, episode_end


class PalletLoadingSim(object):
    def __init__(self, 
                 resolution=512,
                 num_steps=100,
                 num_preview=5,
                 box_norm=False,
                 action_norm=False,
                 render=False,
                 block_size_min=0.2,
                 block_size_max=0.4,
                 discrete_block=False,
                 max_levels=1,
                 show_q=False,
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
        self.use_discrete_block = discrete_block

        self.box_height = 0.15
        self.q_value = None
        self.block_que = []
        self.max_levels = max_levels

        self.state, self.next_block, self.level_map = None, None, None
        self.stacked_history = None

        if self.render:
            self.renderer = Renderer(resolution, show_q)

    def reset(self):
        self.state, self.next_block, self.level_map = self.init_scenario()

        self.stacked_history = {
            "pose_list": [],
            "quat_list": [],
            "scale_list": [],
        }

        if self.render:
            next_blocks = [self.next_block] + self.block_que[:-1]
            self.renderer.render_current_state(self.state, next_blocks, self.state)
        
        next_block = np.array(self.next_block)
        if self.box_norm:
            next_block = self.next_block
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
            
        obs = (self.state, next_block[:2])
        return obs
    
    def init_scenario(self):
        if self.max_levels == 1:
            init_state = np.zeros([self.resolution, self.resolution])
            init_map = np.zeros([self.resolution, self.resolution], dtype=np.uint8)
        elif self.max_levels > 1:
            init_state = np.zeros([self.max_levels, self.resolution, self.resolution])
            init_map = np.zeros([self.resolution, self.resolution], dtype=np.uint8)

        self.step_count = 0

        # next block: (height, width)
        self.block_que = []
        for _ in range(self.num_preview):
            new_block = self.make_new_block()
            self.block_que.append(new_block)

        next_block = self.get_next_block()
        return init_state, next_block, init_map

    def make_new_block(self):
        if self.use_discrete_block:
            new_block = 0.9 * np.random.choice([0.2, 0.3, 0.4, 0.5], 2, True,
                                               p=[0.4, 0.3, 0.2, 0.1])
        else:
            new_block = np.random.uniform(self.block_size_min, self.block_size_max, 2)
        new_block = np.append(new_block, self.box_height)
        return new_block

    def get_next_block(self):
        next_block = self.block_que.pop(0)
        new_block = self.make_new_block()
        self.block_que.append(new_block)
        return next_block

    def step(self, action):
        raise NotImplementedError


class Floor1(PalletLoadingSim):
    def __init__(
        self, 
        resolution=512,
        num_steps=100,
        num_preview=5,
        box_norm=False,
        action_norm=False,
        render=False,
        block_size_min=0.2,
        block_size_max=0.4,
        discrete_block=False,
        max_levels=1,
        show_q=False,
        reward_type='binary',
    ):
        
        super().__init__(
            resolution=resolution,
            num_steps=num_steps,
            num_preview=num_preview,
            box_norm=box_norm,
            action_norm=action_norm,
            render=render,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            discrete_block=discrete_block,
            max_levels=max_levels,
            show_q=show_q,
        )

        self.reward_fuc = RewardFunc(reward_type, max_levels=max_levels)
    
    def step(self, action):
        self.step_count += 1

        # previous state #
        previous_state = self.state

        # denormalize action #
        if self.action_norm:
            action_pos = np.array(action[1:]) * self.resolution
        else:
            action_pos = np.array(action[1:])

        if self.box_norm:
            next_block = np.array(self.next_block) * self.resolution
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        
        action_rot = action[0]
        cy, cx = action_pos

        box_level = 1

        pose_ = [cy/self.resolution, cx/self.resolution, (box_level-0.5)*self.box_height]
        scale_ = [self.next_block[0], self.next_block[1], self.box_height]

        if action_rot==0:
            by, bx, bh = next_block
            quat_ = [0.0, 0.0, 0.0, 1.0]
        elif action_rot==1:
            bx, by, bh = next_block
            quat_ = [0.7071, 0.0, 0.0, 0.7071]

        self.stacked_history["pose_list"].append(pose_)
        self.stacked_history["quat_list"].append(quat_)
        self.stacked_history["scale_list"].append(scale_)

        min_y = np.round(cy - (by-1e-5)/2).astype(int)
        min_x = np.round(cx - (bx-1e-5)/2).astype(int)
        max_y = np.round(cy + (by-1e-5)/2).astype(int)
        max_x = np.round(cx + (bx-1e-5)/2).astype(int)
        next_block_bound = [min_y, max_y, min_x, max_x]

        box_placed = np.zeros(np.shape(self.state))
        box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1   
        self.state = self.state + box_placed

        box_level = np.max(self.level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1
        self.level_map[max(min_y,0):max_y,max(min_x,0):max_x] = box_level

        if self.render:
            next_blocks = self.block_que
            self.renderer.render_current_state(self.state, next_blocks, previous_state,
                                               box=next_block_bound)

        reward, episode_end = self.reward_fuc.get_2d_reward(previous_state, next_block_bound)

        self.next_block = self.get_next_block()
        if self.box_norm:
            next_block = self.next_block
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        
        next_block = np.array(next_block)
        obs = (self.state, next_block[:2])        
        return obs, reward, episode_end
    

class FloorN(PalletLoadingSim):
    def __init__(
        self, 
        resolution=512,
        num_steps=100,
        num_preview=5,
        box_norm=False,
        action_norm=False,
        render=False,
        block_size_min=0.2,
        block_size_max=0.4,
        discrete_block=False,
        max_levels=2,
        show_q=False,
        reward_type='binary',
    ):
        
        super().__init__(
            resolution=resolution,
            num_steps=num_steps,
            num_preview=num_preview,
            box_norm=box_norm,
            action_norm=action_norm,
            render=render,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            discrete_block=discrete_block,
            max_levels=max_levels,
            show_q=show_q,
        )

        assert max_levels >= 2

        from environment.sim_app import StabilityChecker
        stability_checker = StabilityChecker(box_height=self.box_height+6e-3, max_level=5)

        self.reward_fuc = RewardFunc(reward_type,
                                     stability_sim=stability_checker,
                                     max_levels=max_levels,
                                     sim_render=render)

        self.stacked_history = {
            "pose_list": [],
            "quat_list": [],
            "scale_list": [],
        }
    
    def step(self, action):
        self.step_count += 1

        # previous state #
        previous_state = np.copy(self.state)

        # denormalize action #
        if self.action_norm:
            action_pos = np.array(action[1:]) * self.resolution
        else:
            action_pos = np.array(action[1:])

        if self.box_norm:
            next_block = np.array(self.next_block) * self.resolution
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        
        action_rot = action[0]
        cy, cx = action_pos

        if action_rot==0:
            by, bx, _ = next_block
            quat_ = [0.0, 0.0, 0.0, 1.0]

        elif action_rot==1:
            bx, by, _ = next_block
            quat_ = [0.7071, 0.0, 0.0, 0.7071]

        min_y = np.round(cy - (by-1e-5)/2).astype(int)
        min_x = np.round(cx - (bx-1e-5)/2).astype(int)
        max_y = np.round(cy + (by-1e-5)/2).astype(int)
        max_x = np.round(cx + (bx-1e-5)/2).astype(int)
        next_block_bound = [min_y, max_y, min_x, max_x]

        box_placed = np.zeros(np.shape(self.state[0]))
        box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1

        box_level = np.max(self.level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1
        self.level_map[max(min_y,0):max_y,max(min_x,0):max_x] = box_level
        assert box_level >= 1

        if box_level <= self.max_levels:
            self.state[box_level-1] = self.state[box_level-1] + box_placed

        if self.render:
            next_blocks = self.block_que
            self.renderer.render_current_state(self.state, next_blocks, previous_state,
                                               box=next_block_bound)

        pose_ = [cy/self.resolution, cx/self.resolution, (box_level-1)*self.box_height+0.01]
        scale_ = [self.next_block[0], self.next_block[1], self.box_height]

        self.stacked_history["pose_list"].append(pose_)
        self.stacked_history["quat_list"].append(quat_)
        self.stacked_history["scale_list"].append(scale_)

        reward, episode_end = self.reward_fuc.get_3d_reward(previous_state,
                                                            next_block_bound,
                                                            self.stacked_history,
                                                            self.level_map, box_level)


        self.next_block = self.get_next_block()
        if self.box_norm:
            next_block = self.next_block
        else:
            next_block = np.round(np.array(self.next_block) * self.resolution).astype(int)
        
        next_block = np.array(next_block)
        obs = (self.state, next_block[:2])
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
