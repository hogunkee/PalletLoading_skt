# naive mlp model that inputs the state and outputs random position
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from isaac.isaac_utils import *

X_LOW = 0.3
X_HIGH = 1.3
Y_LOW = -0.8
Y_HIGH = 0.2

class NaivePlanner(nn.Module):
    def __init__(self):
        super(NaivePlanner, self).__init__()

    def forward(self):
        x = torch.rand(3).detach().cpu().numpy()
        x[0] = x[0] * (X_HIGH - X_LOW) + X_LOW
        x[1] = x[1] * (Y_HIGH - Y_LOW) + Y_LOW
        x[2] = -0.3
        return x
    
class NaiveRuleBasedPlanner(nn.Module):
    def __init__(self, origin, size):
        super(NaiveRuleBasedPlanner, self).__init__()
        self.origin = origin
        self.size = size
        self.padding = 0.03 #(m)
        self.x_min = self.origin[0] - self.size[0] / 2.0 + self.padding + 0.1436
        self.y_min = self.origin[1] - self.size[1] / 2.0 + self.padding
        self.x_max = self.origin[0] + self.size[0] / 2.0 - self.padding + 0.1436
        self.y_max = self.origin[1] + self.size[1] / 2.0 - self.padding
        #self.bin_list = []
        self.x_high = self.x_min
        self.y_high = self.y_min
        self.z_high = -0.3
        self.max_x_size = 0
        
    def forward(self, bin_size):
        # start from (x_min, y_min), increase y first until not capable, then increase x once and initialize y
        # if increased x until not capable, increase z once and initialize x, y
        new_y_high = self.y_high + bin_size[1] + 2 * self.padding
        if new_y_high > self.y_max:
            self.x_high += self.max_x_size + 2 * self.padding
            self.y_high = self.y_min
            self.max_x_size = 0
            
        new_x_high = self.x_high + bin_size[0] + 2 * self.padding
        if new_x_high > self.x_max:
            self.x_high = self.x_min
            self.y_high = self.y_min
            self.z_high += 0.15
        
        x = self.x_high + bin_size[0] / 2.0 + self.padding
        y = self.y_high + bin_size[1] / 2.0 + self.padding
        z = self.z_high
        self.y_high = y + bin_size[1] / 2.0 + self.padding
        
        if self.max_x_size < bin_size[0]:
            self.max_x_size = bin_size[0]
        
        pos = torch.tensor([x, y, z], dtype=torch.float32).detach().cpu().numpy()
        
        return pos
    
class RuleBasedPlanner(nn.Module):
    def __init__(self, origin, size):
        super(RuleBasedPlanner, self).__init__()
        self.origin = origin
        self.origin_offset = np.array([0.1436, 0.0])
        self.size = size
        self.pallet_padding = 0.05
        self.bin_padding = 0.01 #(m)
        self.x_min = self.origin[0] + self.origin_offset[0] - self.size[0] / 2.0 + self.pallet_padding
        self.y_min = self.origin[1] + self.origin_offset[1] - self.size[1] / 2.0 + self.pallet_padding
        self.x_max = self.origin[0] + self.origin_offset[0] + self.size[0] / 2.0 - self.pallet_padding
        self.y_max = self.origin[1] + self.origin_offset[1] + self.size[1] / 2.0 - self.pallet_padding
        
        self.corner = np.array([self.x_min, self.y_min]) # position of right down corner of new bin.
        
        self.x_high_list = []
        self.y_bound_list = [] # start from self.y_min, but abbreviated.
        self.last_x_high_list = [self.x_min]
        self.last_y_bound_list = [self.y_max] # start from self.y_min, but abbreviated.
        
        self.z = -0.3
        self.z_inc = 0.15
        
    def forward(self, bin_size):
        # if new bin y size is out of bound
        if self.corner[1] + bin_size[1] + 2 * self.bin_padding > self.y_max:            
            for x_high, y_bound in zip(self.last_x_high_list, self.last_y_bound_list):
                if self.corner[1] < y_bound:
                    self.x_high_list.append(x_high)
                    self.y_bound_list.append(y_bound)
            
            self.corner[1] = self.y_min
            
            self.last_x_high_list = self.x_high_list
            self.last_y_bound_list = self.y_bound_list
            self.x_high_list = []
            self.y_bound_list = []               
        
        # calculate current corner[0]
        x_high_max = self.x_min
        for x_high, y_bound in zip(self.last_x_high_list, self.last_y_bound_list):
            if self.corner[1] < y_bound:
                if x_high_max < x_high:
                    x_high_max = x_high
                
                # if bin size is bigger than range, continue search.
                if y_bound - self.corner[1] < bin_size[1]:
                    continue
                
                self.corner[0] = x_high_max
                break
        
        # if new bin x size is out of bound
        if self.corner[0] + bin_size[0] + 2 * self.bin_padding > self.x_max:
            self.corner[0] = self.x_min
            self.corner[1] = self.y_min
            self.z += self.z_inc
            
            self.last_x_high_list = [self.x_min]
            self.last_y_bound_list = [self.y_max]
            self.x_high_list = []
            self.y_bound_list = []
        
        x = self.corner[0] + bin_size[0] / 2.0 + self.bin_padding
        y = self.corner[1] + bin_size[1] / 2.0 + self.bin_padding
        z = self.z
        
        self.x_high_list.append(self.corner[0] + bin_size[0] + 2 * self.bin_padding)
        
        # calculate next corner[1]
        self.corner[1] += bin_size[1] + 2 * self.bin_padding 
        self.y_bound_list.append(self.corner[1])
        
        pos = torch.tensor([x, y, z], dtype=torch.float32).detach().cpu().numpy()
        
        return pos
    
class ModelBasedPlanner(object):
    def __init__(self, origin, size, agent, args):
        self.origin = origin
        self.origin_offset = np.array([0.1436, 0.0])
        self.size = size
        self.pallet_padding = 0.05
        self.bin_padding = 0.01 #(m)
        self.x_min = self.origin[0] + self.origin_offset[0] - self.size[0] / 2.0 + self.pallet_padding
        self.y_min = self.origin[1] + self.origin_offset[1] - self.size[1] / 2.0 + self.pallet_padding
        self.x_max = self.origin[0] + self.origin_offset[0] + self.size[0] / 2.0 - self.pallet_padding
        self.y_max = self.origin[1] + self.origin_offset[1] + self.size[1] / 2.0 - self.pallet_padding
        
        self.corner = np.array([self.x_max, self.y_min]) # position of top left corner of new bin.
        
        self.x_high_list = []
        self.y_bound_list = [] # start from self.y_min, but abbreviated.
        self.last_x_high_list = [self.x_min]
        self.last_y_bound_list = [self.y_max] # start from self.y_min, but abbreviated.
        
        self.z = -0.3
        self.z_inc = 0.15 * args.pallet_scale
        
        self.resolution = args.resolution
        self.max_levels = args.max_levels
        self.box_norm = True
        self.box_height = 0.15 * args.pallet_scale
        
        self.state = np.zeros((args.max_levels, args.resolution, args.resolution))
        self.level_map = np.zeros((args.resolution, args.resolution))
        self.poses = []
        self.quats = []

        self.agent = agent

        self.use_bound_mask = args.use_bound_mask
        self.use_floor_mask = args.use_floor_mask
        self.use_projection = args.use_projection
        self.use_coordnconv = args.use_coordnconv

        self.renderer = Renderer(self.resolution, show_q=False)
        
    def reset(self):
        self.state = np.zeros((self.max_levels, self.resolution, self.resolution))
        self.level_map = np.zeros((self.resolution, self.resolution))
        self.previous_map = np.zeros((self.resolution, self.resolution))
        self.poses = []
        self.quats = []
        
    def get_action(self, block, scale=0.8):
        # change current pallet info to discretized board
        if len(block) == 3:
            block = block[:2]

        block /= scale

        with_q = True
        deterministic = True

        if self.use_projection:
            p_project = 1.000
        else:
            p_project = 0.000

        # generate q_mask
        q_mask = np.ones((2,self.resolution,self.resolution))
        if self.use_bound_mask:
            q_mask = generate_bound_mask(self.state, block[:2])
        if self.use_floor_mask:
            q_mask = generate_floor_mask(self.state, block[:2], q_mask)

        action, _ = self.agent.get_action(self.state, block, q_mask,
                                          deterministic=True,
                                          p_project=p_project)

        self.previous_map = self.level_map.copy()
        pose_, rot_, next_block_bound_ = self.update_state(action, block, scale)
               
        # render
        self.renderer.render_current_state(self.level_map/self.max_levels, block, 
                                           self.previous_map/self.max_levels,
                                           next_block_bound_)

        return pose_, rot_
    
    def update_state(self, action, block, scale=0.8):
        action_rot = action[0]
        cy, cx = np.array(action[1:])        

        if action_rot==0: by, bx = block
        elif action_rot==1: bx, by = block
        by, bx = by * self.resolution, bx * self.resolution

        next_block_bound = get_block_bound(cy, cx, by, bx)
        min_y, max_y, min_x, max_x = next_block_bound

        box_placed = np.zeros(np.shape(self.state[0]))
        box_placed[max(min_y,0): max_y, max(min_x,0): max_x] = 1

        box_level = int(np.max(self.level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1)
        self.level_map[max(min_y,0):max_y,max(min_x,0):max_x] = box_level
        assert box_level >= 1

        if box_level <= self.max_levels:
            self.state[box_level-1] = self.state[box_level-1] + box_placed

        # get pose and scale based on current pallet
        pose_ = np.array([-(min_y+max_y)/2/self.resolution*scale + self.corner[0], 
                 (min_x+max_x)/2/self.resolution*scale + self.corner[1], 
                 -0.4 + ((box_level-1)*self.box_height)*scale])
        
        self.poses.append(pose_)
        if action_rot == 0:
            self.quats.append([0.0, 0.0, 0.0, 1.0])
        elif action_rot == 1:
            self.quats.append([0.7071, 0.0, 0.0, 0.7071])
        return pose_, action_rot, next_block_bound