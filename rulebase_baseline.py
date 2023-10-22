import random
import math
import numpy as np
from environment import Floor1

"""
Ruebase algorithm for pallet packing.

"""


class RulebasePalletLoader:
    def __init__(self, obs_resolution):
        self.obs_resolution = obs_resolution

    def reset(self):
        self.prev_head_i = 0
        self.prev_tail_j = -1
        self.curr_head_i = 0
        self.curr_tail_j = 0

    def get_action_from_pos(self, pos_i, pos_j, block_pixel_width, block_pixel_height):
        # print(pos_i, pos_j, block_pixel_width, block_pixel_height)
        return (
            (pos_i + 0.5 * block_pixel_width) / self.obs_resolution,
            (pos_j + 0.5 * block_pixel_height) / self.obs_resolution,
        )

    def check_if_block_fit_after_prev_tail_j(self, block_dimension_in_pixel):
        block_pixel_width, block_pixel_height = block_dimension_in_pixel
        if self.prev_tail_j < 0:
            return False
        if self.prev_tail_j > self.obs_resolution - block_pixel_height:
            return False
        # TODO:

    def check_if_block_fit_after_curr_tail_j(self, block_dimension_in_pixel):
        block_pixel_width, block_pixel_height = block_dimension_in_pixel
        rotations = [0, 1]

        for rotation in rotations:
            if not rotation:
                block_pixel_width, block_pixel_height = block_dimension_in_pixel
            else:
                block_pixel_height, block_pixel_width = block_dimension_in_pixel
            # lowered = 0
            if self.curr_tail_j > self.obs_resolution - block_pixel_height:
                continue
            lowered = 0  # TODO:
            if self.prev_head_i - lowered + block_pixel_width > self.obs_resolution:
                continue

            self.curr_head_i = max(
                self.curr_head_i, self.prev_head_i - lowered + block_pixel_width
            )
            action_i, action_j = self.get_action_from_pos(
                self.prev_head_i - lowered,
                self.curr_tail_j,
                block_pixel_width,
                block_pixel_height,
            )
            # print("action_i, action_j", action_i, action_j)
            self.curr_tail_j = self.curr_tail_j + block_pixel_height
            return [rotation, action_i, action_j]

        return None

    def update_head_n_tail(self, block_dimension_in_pixel):
        # returns False if block cant' be fit into pallet
        ########## 1. Check if more block can't be fit into pallet ##########
        block_pixel_width, block_pixel_height = block_dimension_in_pixel
        if self.curr_head_i > self.obs_resolution:
            raise Exception("curr_head_i > obs res")
        # TODO: check if curr_tail_j can be moved back
        if (
            self.curr_tail_j <= self.obs_resolution - block_pixel_height
            and self.curr_tail_j <= self.obs_resolution - block_pixel_width
        ):
            lowered = 0  # TODO:
            if (
                self.prev_head_i - lowered + block_pixel_width > self.obs_resolution
                and self.prev_head_i - lowered + block_pixel_height
                > self.obs_resolution
            ):
                return False
        # return False
        ########## 2. Load next row(block can't fit after curr_tail_j) ##########
        if self.curr_head_i == self.obs_resolution:
            return False
        self.prev_tail_j = self.curr_tail_j
        self.prev_head_i = self.curr_head_i
        self.curr_tail_j = 0
        # print("box: ", block_dimension_in_pixel)
        # print("prev curr tail_j", self.prev_tail_j, self.curr_tail_j)
        # print("prev curr head_i", self.prev_head_i, self.curr_head_i)
        return True

    def get_action(self, obs):
        image_obs, block_size = obs
        block_pixel_dim = (
            math.ceil((block_size[0] / 0.9 - 1e-3) * self.obs_resolution),
            math.ceil((block_size[1] / 0.9 - 1e-3) * self.obs_resolution),
        )
        # print("block size:", block_size)
        while True:
            # action = self.check_if_block_fit_after_prev_tail_j()
            # if action != None:
            #     break

            action = self.check_if_block_fit_after_curr_tail_j(block_pixel_dim)
            # print("after curr tail:", action)
            if action != None:
                break

            if not self.update_head_n_tail(block_pixel_dim):
                # print("end ep")
                action = [0, 0, 0]
                break

        return action


if __name__ == "__main__":
    box_norm = True
    resolution = 64
    env = Floor1(
        resolution=resolution,
        box_norm=box_norm,
        action_norm=True,
        render=False,
        discrete_block=True,
    )
    predictor = RulebasePalletLoader(resolution)

    total_reward = 0.0
    num_episodes = 1000
    for ep in range(num_episodes):
        obs = env.reset()
        predictor.reset()
        ep_reward = 0.0
        # print(f'Episode {ep} starts.')
        for i in range(100):
            # print(obs[0])
            action = predictor.get_action(obs)
            obs, reward, end = env.step(action)
            ep_reward += reward
            if end:
                # print('Episode ends.')
                break
        # print("    ep_reward: ", ep_reward)
        total_reward += ep_reward
    avg_score = total_reward / num_episodes
    print("average score: ", avg_score)
