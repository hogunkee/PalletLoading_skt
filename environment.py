import time
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from typing import List


def is_out_of_range(x, y, width, height):
    if (
        x - width * 0.5 < 0
        or y - height * 0.5 < 0
        or x + width * 0.5 > 1
        or y + height * 0.5 > 1
    ):
        return True
    return False


class Block:
    counter = 0

    def __init__(self, x, y, width, height):
        self.id = Block.counter
        self.x_range = [x - width * 0.5, x + width * 0.5]
        self.y_range = [y - height * 0.5, y + height * 0.5]

    def is_overlap(self, target_block):
        if (
            min(self.x_range[1], target_block.x_range[1])
            > max(self.x_range[0], target_block.x_range[0])
        ) and (
            min(self.y_range[1], target_block.y_range[1])
            > max(self.y_range[0], target_block.y_range[0])
        ):
            return True
        return False


class Floor:
    def __init__(self, floor):
        self.floor = floor
        self.blocks: List[Block] = []
        self.num_blocks = 0

    def reset(self):
        self.num_blocks = 0
        self.blocks = []

    def is_overlap(self, new_block: Block):
        for old_block in self.blocks:
            if new_block.is_overlap(old_block):
                return True
        return False

    def load(self, block: Block):
        if self.is_overlap(block):
            # Fail to load block
            return False
        self.blocks.append(block)
        self.num_blocks += 1
        return True


class PalletLoading(object):
    def __init__(
        self,
        obs_resolution=32,
        render_resolution=512,
        num_steps=100,
        num_preview=5,
        box_norm=False,
        action_norm=False,
        render=False,
        block_size_min=0.2,
        block_size_max=0.4,
        plot_obs=True,
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

        render: bool (default: False)
            rendering the current palette state.

        plot_obs: bool (default: True)
            render image observation
        """
        self.obs_resolution = obs_resolution
        self.render_resolution = render_resolution
        self.num_steps = num_steps
        self.num_preview = num_preview
        self.box_norm = box_norm
        self.action_norm = action_norm
        self.render = render
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max
        self.plot_obs = plot_obs

        if self.render:
            plot0 = plt.subplot2grid((2, 3), (0, 0))
            plot1 = plt.subplot2grid((2, 3), (0, 1))
            plot2 = plt.subplot2grid((2, 3), (0, 2))
            plot3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
            if self.plot_obs:
                plot0.set_title(f"obs_img ({obs_resolution}x{obs_resolution})")
            else:
                plot0.set_title("previous state")
            plot1.set_title("current state")
            plot2.set_title("next block")
            plot3.set_title("next blocks")
            self.plots = [plot0, plot1, plot2, plot3]
            for i, p in enumerate(self.plots):
                p.set_xticks([])
                p.set_yticks([])
                if i == 0 or i == 1:
                    p.axis("off")
            plt.show(block=False)

    def reset(self):
        self.floor = Floor(floor=1)
        Block.counter = 0
        self.obs_img, self.render_state, self.next_block = self.init_scenario()
        if self.render:
            self.render_current_state(self.render_state)

        next_block = np.array(self.next_block)
        obs = (self.obs_img, next_block)
        return obs
        # return self.render_state, self.next_block

    def render_current_state(self, previous_render_state=None, box=None):
        pad = int(0.1 * self.render_resolution)
        render_state_pad = (
            np.ones(
                [
                    int(1.2 * self.render_resolution),
                    int(1.2 * self.render_resolution),
                    3,
                ]
            )
            * 0.7
        )
        pad_mask = np.ones(render_state_pad.shape[:2])
        pad_mask[pad:-pad, pad:-pad] = 0

        if self.plot_obs:
            obs_pad = 1
            image_obs_pad = (
                np.ones(
                    [
                        int(self.obs_resolution + obs_pad * 2),
                        int(self.obs_resolution + obs_pad * 2),
                        3,
                    ]
                )
                * 0.7
            )
            y, x = np.where(self.obs_img == 0)
            image_obs_pad[y + obs_pad, x + obs_pad] = (1, 1, 1)
            y, x = np.where(self.obs_img == 1)
            image_obs_pad[y + obs_pad, x + obs_pad] = (0, 0, 0)
            y, x = np.where(self.obs_img >= 2)
            image_obs_pad[y + obs_pad, x + obs_pad] = (1, 0, 0)
            self.plots[0].imshow(image_obs_pad)
        else:
            pre_state_pad = render_state_pad.copy()
            y, x = np.where(previous_render_state == 0)
            pre_state_pad[y + pad, x + pad] = (1, 1, 1)
            y, x = np.where(previous_render_state == 1)
            pre_state_pad[y + pad, x + pad] = (0, 0, 0)
            y, x = np.where(previous_render_state == 2)
            pre_state_pad[y + pad, x + pad] = (1, 0, 0)
            y, x = np.where(
                np.all(pre_state_pad != [0.7, 0.7, 0.7], axis=-1) & (pad_mask == 1)
            )
            pre_state_pad[y, x] = (1, 0, 0)
            self.plots[0].imshow(pre_state_pad)

        # render_state_pad = np.ones([int(1.2*self.render_resolution), int(1.2*self.render_resolution), 3]) * 0.7
        y, x = np.where(self.render_state == 0)
        render_state_pad[y + pad, x + pad] = (1, 1, 1)
        y, x = np.where(self.render_state == 1)
        render_state_pad[y + pad, x + pad] = (0, 0, 0)
        if box is not None:
            min_y, max_y, min_x, max_x = np.array(box) + pad
            render_state_pad[min_y:max_y, min_x:max_x] = (0, 0, 1)
        y, x = np.where(self.render_state == 2)
        render_state_pad[y + pad, x + pad] = (1, 0, 0)
        y, x = np.where(
            np.all(render_state_pad != [0.7, 0.7, 0.7], axis=-1) & (pad_mask == 1)
        )
        render_state_pad[y, x] = (1, 0, 0)
        self.plots[1].imshow(render_state_pad)

        block_figures = None
        if self.step_count == 0:
            next_blocks = [self.next_block] + self.block_que[:-1]
        else:
            next_blocks = self.block_que
        # for i, b in enumerate([self.next_block] + self.block_que[:-1]):
        for i, b in enumerate(next_blocks):
            cy, cx = int(0.6 * self.render_resolution), int(
                0.6 * self.render_resolution
            )
            b = np.round(np.array(b) * self.render_resolution).astype(int)
            by, bx = b
            min_y = np.round(cy - (by - 1e-5) / 2).astype(int)
            min_x = np.round(cx - (bx - 1e-5) / 2).astype(int)
            max_y = np.round(cy + (by - 1e-5) / 2).astype(int)
            max_x = np.round(cx + (bx - 1e-5) / 2).astype(int)

            block_fig = np.ones_like(render_state_pad)
            block_fig[min_y:max_y, min_x:max_x] = [0, 0, 0]

            if i == 0:
                self.plots[2].imshow(block_fig)
            else:
                if block_figures is None:
                    block_figures = block_fig
                else:
                    block_figures = np.concatenate([block_figures, block_fig], axis=1)
        self.plots[3].imshow(block_figures)
        plt.draw()
        # plt.pause(0.01)
        plt.pause(1)

    def get_next_block(self):
        next_block = self.block_que.pop(0)
        new_block = np.random.uniform(self.block_size_min, self.block_size_max, 2)
        self.block_que.append(new_block)
        return next_block

    def init_scenario(self):
        obs_img = np.zeros([self.obs_resolution, self.obs_resolution])
        render_state = np.zeros([self.render_resolution, self.render_resolution])
        self.step_count = 0

        # next block: (height, width)
        self.block_que = np.random.uniform(
            self.block_size_min, self.block_size_max, [self.num_preview, 2]
        ).tolist()

        next_block = self.get_next_block()
        return obs_img, render_state, next_block

    def place_block_in_render_state(self, normalized_action):
        action = np.round(np.array(normalized_action) * self.render_resolution)
        # make box region #
        cy, cx = action
        by, bx = np.round(np.array(self.next_block_rotated)*self.render_resolution).astype(int)
        min_y = np.round(cy - (by - 1e-5) / 2).astype(int)
        min_x = np.round(cx - (bx - 1e-5) / 2).astype(int)
        max_y = np.round(cy + (by - 1e-5) / 2).astype(int)
        max_x = np.round(cx + (bx - 1e-5) / 2).astype(int)

        box_placed = np.zeros([self.render_resolution, self.render_resolution])
        box_placed[min_y:max_y, min_x:max_x] = 1
        self.render_state = self.render_state + box_placed
        self.render_box = [min_y, max_y, min_x, max_x]

    def place_block_in_obs_img(self, normalized_action):
        action = np.round(np.array(normalized_action) * self.obs_resolution)
        # make box region #
        cy, cx = action
        by, bx = np.round(np.array(self.next_block_rotated)*self.render_resolution).astype(int)
        min_y = np.floor(cy - (by + 1e-5) / 2).astype(int)
        min_x = np.floor(cx - (bx + 1e-5) / 2).astype(int)
        max_y = np.floor(cy + (by + 1e-5) / 2).astype(int)
        max_x = np.floor(cx + (bx + 1e-5) / 2).astype(int)

        box_placed = np.zeros([self.obs_resolution, self.obs_resolution])
        box_placed[min_y : max_y, min_x : max_x] = 1
        self.obs_img = self.obs_img + box_placed

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        out_of_range = False
        collision = False
        episode_end = False

        # previous state #
        previous_render_state = self.render_state

        # check if action contains transformation
        if(len(action) == 3):
            action_pos = action[:2]
            action_rot = action[2]
        elif(len(action)==2):
            action_pos = action
            action_rot = 0
        else:
            raise Exception("Action space should be [p_x, p_y] or [p_x, p_y, roatation \in {0,1}]")
        
        # denormalize the action
        if not self.action_norm:
            action_pos = np.array(action_pos) / self.render_resolution

        # rotate block by an action
        if action_rot:
            self.next_block_rotated = np.array([self.next_block[1], self.next_block[0]])
        else:
            self.next_block_rotated = self.next_block

        # clip action to (0.0, 1.0)
        action_pos = np.clip(action_pos, 0.0, 1.0)
        if self.render:
            self.place_block_in_render_state(action_pos)
        self.place_block_in_obs_img(action_pos)

        # check out of range #
        y_action, x_action = action_pos
        block_height, block_width = self.next_block_rotated

        if is_out_of_range(x_action, y_action, block_width, block_height):
            out_of_range = True

        # check collision #
        new_block = Block(x_action, y_action, block_width, block_height)
        #if not self.floor.load(new_block):
        if len(np.where(self.obs_img>1)[0]) > 0:
            collision = True

        # if no OOR of collision, the placement succeeds #
        if not (out_of_range or collision):
            reward = 1.0
        else:
            episode_end = True

        if self.render:
            self.render_current_state(previous_render_state, self.render_box)

        self.next_block = self.get_next_block()
        if self.box_norm:
            next_block = self.next_block
        else:
            next_block = np.round(
                np.array(self.next_block) * self.render_resolution
            ).astype(int)

        next_block = np.array(next_block)
        obs = (self.obs_img, next_block)
        # return state, next_block, reward, episode_end
        return obs, reward, episode_end


if __name__ == "__main__":
    import random

    box_norm = True
    env = PalletLoading(
        obs_resolution=10,
        box_norm=box_norm,
        render=True,
        block_size_min=0.1,
        block_size_max=0.25,
    )
    # state, next_block = env.reset()

    total_reward = 0.0
    num_episodes = 100
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        # print(f'Episode {ep} starts.')
        for i in range(100):
            random_action = np.random.uniform(0.1, 0.9, 2)
            action = random_action.tolist()
            # random rotation
            action_rot = random.randrange(0,2)
            action.append(action_rot)
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
