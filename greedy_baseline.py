import random
import math
import numpy as np
from environment import Floor1


class RulebasePallasdfetLoader:
    def __init__(self, obs_resolution, margin_ratio=0.05):
        self.obs_resolution = obs_resolution
        self.margin_ratio = margin_ratio

    def greedy_search(self, image_obs, block_size):
        # print(image_obs)
        block_res_width = math.ceil(block_size[0] * self.obs_resolution)
        block_res_height = math.ceil(block_size[1] * self.obs_resolution)
        # print("block res", block_res_width, block_res_height)
        for elem in range(self.obs_resolution * self.obs_resolution):
            obs_i = elem // self.obs_resolution
            obs_j = elem % self.obs_resolution

            # check out of bound
            block_out_of_index = False
            if (
                obs_i + block_res_width > self.obs_resolution
                or obs_j + block_res_height > self.obs_resolution
            ):
                block_out_of_index = True

            if block_out_of_index:
                # print("OoB", obs_i, obs_j)
                continue
            # check collision
            block_collision = False
            for block_pix in range(block_res_width * block_res_height):
                d_i = (
                    block_pix // block_res_height
                )  # Integer division to get the current row
                d_j = block_pix % block_res_height  # Modulus to get the current column
                if image_obs[obs_i + d_i][obs_j + d_j] == 1:
                    block_collision = True
                    # print("COLLS", obs_i + d_i, obs_j + d_j)
                    break
            if not block_collision:
                # found valid action!
                # print("action found:", obs_i, obs_j)
                action_i = (obs_i + 0.5 * block_res_width) / self.obs_resolution
                action_j = (obs_j + 0.5 * block_res_height) / self.obs_resolution
                # print("\t\treturn:", action_i, action_j)
                return [action_i, action_j]

        # print("action not found")
        return None

    def rotate_block(self, block_size):
        rotated_block = [block_size[1], block_size[0]]
        return rotated_block

    def get_greedy_action(self, obs):
        image_obs, block_size = obs
        action_pos = self.greedy_search(image_obs, block_size)
        action_rot = [0]
        if action_pos == None:
            action_pos = self.greedy_search(image_obs, self.rotate_block(block_size))
            action_rot = [1]

        if action_pos == None:
            return [0, 0, 0]
        return action_rot + action_pos


if __name__ == "__main__":
    box_norm = True
    resolution = 10
    env = Floor1(
        resolution=resolution,
        box_norm=box_norm,
        action_norm=True,
        render=False,
        discrete_block=True,
    )
    # state, next_block = env.reset()
    predictor = RulebasePallasdfetLoader(resolution)

    total_reward = 0.0
    num_episodes = 100
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        # print(f'Episode {ep} starts.')
        for i in range(100):
            action = predictor.get_greedy_action(obs)
            obs, reward, end = env.step(action)
            ep_reward += reward
            if end:
                # print('Episode ends.')
                break
        # print("    ep_reward: ", ep_reward)
        total_reward += ep_reward
    avg_score = total_reward / num_episodes
    print("average score: ", avg_score)
