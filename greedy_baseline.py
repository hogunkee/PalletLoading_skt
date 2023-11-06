from environment.environment import FloorN as FloorEnv
import math
import numpy as np


def rotate_block(block_size):
    return [block_size[1], block_size[0]]


class GreedyPolicyAgent:
    def __init__(
        self,
        obs_resolution,
        max_level,
        norm_margin_padding=0.01,
    ):
        self.obs_resolution = obs_resolution
        self.norm_margin_padding = norm_margin_padding
        self.max_level = max_level

    def reset(self):
        pass

    def check_placeability(
        self, search_level, image_obs_3d, pos_index, block_dimension_in_pixel
    ):
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

        for block_pix in range(block_pixel_width * block_pixel_height):
            d_i = block_pix // block_pixel_height
            d_j = block_pix % block_pixel_height
            # check collision
            if image_obs_3d[search_level][obs_i + d_i][obs_j + d_j] == 1:
                return False
            # check if gound is solid
            if (
                search_level != 0
                and image_obs_3d[search_level - 1][obs_i + d_i][obs_j + d_j] == 0
            ):
                return False

        return True

    def get_action_from_pos(self, pos_idx, block_pixel_dim, rotation):
        return [
            rotation,
            (pos_idx[0] + 0.5 * block_pixel_dim[0]) / self.obs_resolution,
            (pos_idx[1] + 0.5 * block_pixel_dim[1]) / self.obs_resolution,
        ]

    def get_greedy_n_th_floor_action(
        self, search_level, image_obs_3d, block_pixel_dim, block_pixel_dim_rot
    ):
        for elem in range(self.obs_resolution * self.obs_resolution):
            pos_idx = (elem // self.obs_resolution, elem % self.obs_resolution)

            if self.check_placeability(
                search_level, image_obs_3d, pos_idx, block_pixel_dim
            ):
                return self.get_action_from_pos(pos_idx, block_pixel_dim, 0)
            elif self.check_placeability(
                search_level, image_obs_3d, pos_idx, block_pixel_dim_rot
            ):
                return self.get_action_from_pos(pos_idx, block_pixel_dim_rot, 1)
        return None

    def greedy_search(self, image_obs_3d, block_size):
        block_pixel_width = math.ceil(block_size[0] * self.obs_resolution)
        block_pixel_height = math.ceil(block_size[1] * self.obs_resolution)
        block_pixel_dim = (block_pixel_width, block_pixel_height)
        block_pixel_dim_rot = (block_pixel_height, block_pixel_width)

        for search_level in range(self.max_level):
            action = self.get_greedy_n_th_floor_action(
                search_level, image_obs_3d, block_pixel_dim, block_pixel_dim_rot
            )
            if action != None:
                print("state")
                print(image_obs_3d[search_level])
                if search_level != 0:
                    print("prev state")
                    print(image_obs_3d[search_level - 1])
                print(
                    "level:",
                    search_level,
                    "\taction:",
                    action,
                    "\tblock_size w margin:",
                    block_size,
                )
                return action
        return None

    def get_action_with_margin(self, image_obs_3d, block_size, margin_ratio):
        block_size_with_margin = (
            block_size[0] / margin_ratio,
            block_size[1] / margin_ratio,
        )
        action = self.greedy_search(image_obs_3d, block_size_with_margin)
        return action

    def get_action(self, image_obs_3d, block_size):
        action = self.get_action_with_margin(image_obs_3d, block_size, 0.9)
        if action == None:
            action = self.get_action_with_margin(image_obs_3d, block_size, 1.0)
        if action == None:
            return [0, 0, 0]
        return action


if __name__ == "__main__":
    resolution = 20
    max_level = 5
    reward_type = "dense"
    env = FloorEnv(
        resolution=resolution,
        num_preview=5,
        box_norm=True,
        action_norm=True,
        render=True,
        discrete_block=True,
        max_levels=max_level,
        reward_type="binary",
        # reward_type=reward_type,
    )
    # predictor = GreedyGroundFloorPolicy(resolution, margin_ratio=0.01)
    predictor = GreedyPolicyAgent(
        resolution, max_level=max_level, norm_margin_padding=0.01
    )

    total_reward = 0.0
    num_episodes = 1
    for ep in range(num_episodes):
        obs = env.reset()
        state, block = obs
        # print("state:",state)
        ep_reward = 0.0
        # print(f'Episode {ep} starts.')
        for i in range(100):
            action = predictor.get_action(image_obs_3d=state, block_size=block)
            # print("state[0]:", state[0])
            print("block:", block)
            print("action:", action)
            obs, reward, done = env.step(action)
            print("done", done)
            state, block = obs
            ep_reward += reward
            if done:
                # print('Episode ends.')
                break
        # print("    ep_reward: ", ep_reward)
        total_reward += ep_reward
        if ep % 10 == 0:
            print(ep + 1, "/", num_episodes, "avg_score:", total_reward / (ep + 1))
    avg_score = total_reward / num_episodes
    print("average score: ", avg_score)
