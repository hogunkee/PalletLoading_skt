import random
import math
import numpy as np
from environment import Floor1


class GreedyPolicyPalletLoader:
    def __init__(self, obs_resolution):
        self.obs_resolution = obs_resolution

    def check_if_block_fits(self, image_obs, pos_index, block_dimension_in_pixel):
        # Time complexity of this method can be reduced to O(1) from current O(n^2) using memory (if min width of the box is set)!
        # Return true if give block fits in given position(index) of the pallet

        # position index of intrerest in observation
        obs_i, obs_j = pos_index
        block_pixel_width, block_pixel_height = block_dimension_in_pixel

        # check out of bound
        if (
            obs_i + block_pixel_width > self.obs_resolution
            or obs_j + block_pixel_height > self.obs_resolution
        ):
            return False
        # check collision
        for block_pix in range(block_pixel_width * block_pixel_height):
            d_i = block_pix // block_pixel_height
            d_j = block_pix % block_pixel_height
            if image_obs[obs_i + d_i][obs_j + d_j] == 1:
                return False
        return True

    def greedy_search(self, image_obs, block_size):
        block_pixel_width = math.ceil(block_size[0] * self.obs_resolution)
        block_pixel_height = math.ceil(block_size[1] * self.obs_resolution)
        for elem in range(self.obs_resolution * self.obs_resolution):
            pos_idx = (elem // self.obs_resolution, elem % self.obs_resolution)
            block_pixel_dim = (block_pixel_width, block_pixel_height)
            block_pixel_dim_rot = (block_pixel_height, block_pixel_width)

            if self.check_if_block_fits(image_obs, pos_idx, block_pixel_dim):
                # found valid action!
                action_i = (pos_idx[0] + 0.5 * block_pixel_dim[0]) / self.obs_resolution
                action_j = (pos_idx[1] + 0.5 * block_pixel_dim[1]) / self.obs_resolution
                return [0, action_i, action_j]
            elif self.check_if_block_fits(image_obs, pos_idx, block_pixel_dim_rot):
                # found valid action!
                action_i = (
                    pos_idx[0] + 0.5 * block_pixel_dim_rot[0]
                ) / self.obs_resolution
                action_j = (
                    pos_idx[1] + 0.5 * block_pixel_dim_rot[1]
                ) / self.obs_resolution
                return [1, action_i, action_j]
        return None

    def rotate_block(self, block_size):
        rotated_block = [block_size[1], block_size[0]]
        return rotated_block

    def get_action(self, obs):
        image_obs, block_size = obs
        block_size_with_margin = (
            block_size[0] / 0.9 - 1e-3,
            block_size[1] / 0.9 - 1e-3,
        )
        action = self.greedy_search(image_obs, block_size_with_margin)
        if action == None:
            return [0, 0, 0]
        return action


if __name__ == "__main__":
    box_norm = True
    resolution = 100
    env = Floor1(
        resolution=resolution,
        box_norm=box_norm,
        action_norm=True,
        render=False,
        discrete_block=True,
    )
    predictor = GreedyPolicyPalletLoader(resolution)

    total_reward = 0.0
    num_episodes = 100
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        # print(f'Episode {ep} starts.')
        for i in range(100):
            # print(obs)
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
