# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import scipy.spatial.transform as tf
from scipy.spatial.transform import Rotation
import os
import torch
import time
import math

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )


"""
Main
"""



class StabilityChecker():
    def __init__(self, box_height, max_level=5,
                 pallet_size=1.1, pallet_margin=0.05, pallet_gridsize=0.05):

        #self.simulation_app = SimulationApp(config)

        # set the simulation
        self.sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cpu")
        set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])
        design_scene()

        self.spawn_interval, self.spawn_offset_z = 10, 0.01
        self.box_height, self.max_level = box_height, max_level

        # add boxes
        #self.pallet_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
        self.pallet_usd_path = os.getcwd() + "/environment/sim_objects/Box1.usd"
        self.box_usd_path = os.getcwd() + "/environment/sim_objects/Box0.usd"
        self.box_default_scale = [1.0, 1.0, 1.0]

        pallet_pos = [pallet_size/2.0-pallet_margin-pallet_gridsize/2.0, pallet_size/2.0-pallet_margin, 0.0]
        prim_utils.create_prim(f"/World/Pallet",
                               usd_path=self.pallet_usd_path,
                               position=pallet_pos,
                               orientation=[0.0, 0.0, 0.0, 1.0],
                               scale=[pallet_size, pallet_size, 0.01])
        
        # initialize the simulator
        self.n_boxes = 0        
        self.sim.reset()
        print("[INFO]: Setup complete...")

        self.n_resets = 0


    def reset_sim(self):
        for box_idx in range(self.n_boxes):
            prim_utils.delete_prim(f"/World/Objects/Box{box_idx}")
        self.n_boxes = 0

        self.n_resets += 1
        soft_ = True
        if self.n_resets % 10000 == 0:
            soft_ = False
        self.sim.reset(soft=soft_)

    def set_box_height(self, box_height):
        self.box_height = box_height

    def get_box_position(self, n_boxes=-1):
        box_pose_list, box_quat_list = [], []
        if n_boxes < 0: n_boxes = self.n_boxes

        for i in range(n_boxes):
            box00 = XFormPrim(prim_path=f"/World/Objects/Box{i}")
            pose, quat = box00.get_world_pose()
            box_pose_list.append(pose.tolist())
            box_quat_list.append(quat.tolist())

        return box_pose_list, box_quat_list

    def get_box_level(self, pose_list):
        # find each block level (0~max_level-1)
        level_list, eps = [], 0.02
        for pose in pose_list:
            current_level = int((pose[2]+eps)/self.box_height)
            #assert 0 <=current_level < self.max_level
            level_list.append(current_level)
        return level_list
    
    def arrange_stack_order(self, pose_list, last_action=False):
        if last_action:
            pose_list = pose_list[:-1]
        level_list = self.get_box_level(pose_list)
        # sort stack-order by level
        stack_order = []
        for l in range(self.max_level):
            for i, level in enumerate(level_list):
                if l==level: stack_order.append(i)
        assert len(stack_order) == len(pose_list)

        if last_action:
            stack_order.append(len(stack_order))
        return stack_order

    def check_stability(self, input_list, output_list,
                        min_pose_error=0.03, min_quat_error=10.0, print_info=False):
        input_pose, input_quat = input_list
        output_pose, output_quat = output_list
        n_boxes = len(input_pose)

        input_level = self.get_box_level(input_pose)
        output_level = self.get_box_level(output_pose)

        stability_ = True

        # conditions for position
        moved_boxes = []
        for i in range(n_boxes):
            dist_ = [(a-b)**2 for (a, b) in zip(
                        input_pose[i][:2], output_pose[i][:2]
                    )]
            dist_ = math.sqrt(sum(dist_))

            rot_input_ = Rotation.from_quat(input_quat[i])
            euler_input_ = rot_input_.as_euler('xyz', degrees=True)
            rot_output_ = Rotation.from_quat(output_quat[i])
            euler_output_ = rot_output_.as_euler('xyz', degrees=True)
            euler_ = max([
                abs(euler_input_[0]-euler_output_[0]),
                abs(euler_input_[1]-euler_output_[1]),
                abs(euler_input_[2]-euler_output_[2]),
            ])
            #euler_ = abs(euler_input_[2]-euler_output_[2])
            euler_ = min(euler_, 180-euler_)

            if dist_ > min_pose_error or input_level[i] != output_level[i] or euler_ > min_quat_error:
                stability_ = False
                moved_boxes.append(i)

                # if dist_ > min_pose_error: print("D:", i, dist_)
                # if input_level[i] != output_level[i]: print("L:", i, input_level[i], output_level[i])
                # if euler_ > min_quat_error: print("Q:", i, euler_input_[0]-euler_output_[0], euler_input_[1]-euler_output_[1], euler_input_[2]-euler_output_[2])

        if print_info:
            if len(moved_boxes) == 1:
                print("  * [Unstable] Box[{}] is Moved.".format(moved_boxes))
            elif len(moved_boxes) >= 2:
                print("  * [Unstable] Box[{}] are Moved.".format(moved_boxes))

        return stability_


    def spawn_box(self, pose, quat, scale):
        prim_utils.create_prim(f"/World/Objects/Box{self.n_boxes}",
                               usd_path=self.box_usd_path,
                               position=pose,
                               orientation=quat,
                               scale=scale)
        # if self.n_boxes == 0:
        #     cache = create_bbox_cache(use_extents_hint=False)
        #     boundbox = compute_aabb(cache, "/World/Objects/Box0", include_children=True)
        #     box_height = boundbox[5] - boundbox[2]
        #     self.set_box_height(box_height)

        self.n_boxes += 1
    
    def stacking(self, position_list, quanterinon_list, scale_list=None,
                 render=False, stack_type="in_order"):
        n_boxes = len(position_list)
        assert len(position_list) == len(quanterinon_list)
        if scale_list is None:
            scale_list = [[1.0 for _ in range(3)] for _ in range(n_boxes)]
        assert len(position_list) == len(quanterinon_list) == len(scale_list)

        stack_order = self.arrange_stack_order(position_list, last_action=True)
        position_list = [position_list[i] for i in stack_order]
        quanterinon_list = [quanterinon_list[i] for i in stack_order]
        scale_list = [scale_list[i] for i in stack_order]
            
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        sim_time, count = 0.0, 0

        self.reset_sim()
        box_idx = 0

        if stack_type=="in_order":
            max_count = n_boxes * self.spawn_interval + 50

        elif stack_type=="at_once":
            max_count = n_boxes + 50

            for _ in range(n_boxes-1):
                pose_ = [a+b for (a, b) in zip(
                    position_list[box_idx], [0.0,0.0,self.spawn_offset_z]
                )]
                scale_ = [a*b for (a, b) in zip(
                    self.box_default_scale, scale_list[box_idx]
                )]
                self.spawn_box(pose=pose_, 
                               quat=quanterinon_list[box_idx],
                               scale=scale_)
                box_idx += 1

        # Simulate physics
        while simulation_app.is_running():
            # If simulation is stopped, then exit.
            if self.sim.is_stopped() or count==max_count:
                break
            # If simulation is paused, then skip.
            if not self.sim.is_playing():
                self.sim.step(render=not args_cli.headless)
                continue

            if count % self.spawn_interval == 0 and box_idx < n_boxes:
                pose_ = [a+b for (a, b) in zip(
                    position_list[box_idx], [0.0,0.0,self.spawn_offset_z]
                )]
                scale_ = [a*b for (a, b) in zip(
                    self.box_default_scale, scale_list[box_idx]
                )]
                self.spawn_box(pose=pose_, 
                               quat=quanterinon_list[box_idx],
                               scale=scale_)
                box_idx += 1

            # perform step
            self.sim.step(render=render)
            # update sim-time
            sim_time += sim_dt
            count += 1

        box_inputs = [position_list, quanterinon_list]
        box_outputs = self.get_box_position()
        return self.check_stability(box_inputs, box_outputs)
    
    

def main():
    box_height = 0.156
    checker = StabilityChecker(box_height=box_height, max_level=5)
    render = True

    # stable case
    pose_ex1 = [0.00, 0.00, 0.0*box_height+0.00] #-0.8*box_height]
    pose_ex2 = [0.50, 0.00, 0.0*box_height+0.00] #-0.8*box_height]
    pose_ex3 = [0.25, 0.00, 1.0*box_height+0.00] #-0.8*box_height]
    pose_ex4 = [0.45, 0.00, 2.0*box_height+0.00] #-0.8*box_height]
    pose_ex5 = [0.05, 0.00, 2.0*box_height+0.00] #-0.8*box_height]
    pose_ex = [pose_ex1, pose_ex2, pose_ex3, pose_ex4, pose_ex5]

    quat_0 = [0.0, 0.0, 0.0, 1.0]
    quat_90 = [0.7071, 0.0, 0.0, 0.7071]
    quat_ex = [quat_0, quat_0, quat_90, quat_0, quat_0]

    scale_ex0 = [0.40, 0.40, 0.15] # [30, 30, 15]
    scale_ex1 = [0.30, 0.90, 0.15] # [20, 50, 15]
    scale_ex2 = [0.30, 0.30, 0.15] # [30, 40, 15]
    scale_ex = [scale_ex0, scale_ex0, scale_ex1, scale_ex2, scale_ex2]

    for _ in range(3):
        s_time = time.time()
        stability_ = checker.stacking(pose_ex, quat_ex, scale_ex, render=render)
        if stability_:
            print("[CHECKER] This Structure is Stable.")
        else:
            print("[CHECKER] This Structure is Untable.")
        e_time = time.time()
        print("Time: {:.3f}s".format(e_time-s_time))


    # unstable case
    pose_ex1 = [0.00, 0.00, 1.0*box_height-0.5*box_height]
    pose_ex2 = [0.10, 0.00, 2.0*box_height-0.5*box_height]
    pose_ex3 = [0.20, 0.00, 3.0*box_height-0.5*box_height]
    pose_ex4 = [0.30, 0.00, 4.0*box_height-0.5*box_height]
    pose_ex5 = [0.40, 0.00, 5.0*box_height-0.5*box_height]
    pose_ex = [pose_ex1, pose_ex2, pose_ex3, pose_ex4, pose_ex5]

    quat_0 = [0.0, 0.0, 0.0, 1.0]
    quat_90 = [0.7071, 0.0, 0.0, 0.7071]
    quat_ex = [quat_0, quat_0, quat_90, quat_0, quat_90]

    scale_ex0 = [0.30, 0.30, 0.15]
    scale_ex1 = [0.30, 0.30, 0.15]
    scale_ex2 = [0.30, 0.30, 0.15]
    scale_ex = [scale_ex0, scale_ex1, scale_ex2, scale_ex1, scale_ex2]

    for _ in range(3):
        s_time = time.time()
        stability_ = checker.stacking(pose_ex, quat_ex, scale_ex, render=render)
        if stability_:
            print("[CHECKER] This Structure is Stable.")
        else:
            print("[CHECKER] This Structure is Untable.")
        e_time = time.time()
        print("Time: {:.3f}s".format(e_time-s_time))

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
