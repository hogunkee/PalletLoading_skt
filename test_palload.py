# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import random

from omni.isaac.core.objects import VisualCapsule, VisualSphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.orbit.utils.kit as kit_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

# use custom CortexWorld class
from isaac.custom_classes.cortexrigidprim import CustomCortexRigidPrim
from isaac.custom_classes.cortex_world import CortexWorld
from omni.isaac.cortex.robot import CortexUr10
import omni.isaac.cortex.math_util as math_util
from omni.isaac.cortex.cortex_utils import get_assets_root_path_or_die

import isaac.behaviors.ur10.bin_stacking_behavior as behavior

import os
from isaac.isaac_utils import bbox_info

# relative path cause error (can't load multiple objects from single usd file)
ASSET_DIR = os.getcwd() + "/isaac/assets"

class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path_or_die()

        self.ur10_table_usd = ASSET_DIR + "/ur10_bin_stacking_short_suction.usd"
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.rubiks_cube_usd = self.assets_root_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"
        self.ycb_usd = self.assets_root_path + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
        self.dex_usd = self.assets_root_path + "/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        self.box_usd = ASSET_DIR + "/Box0.usd"
    
def get_scale(prim_path, bin_size):
    range_size = bbox_info(prim_path)
    bin_size = np.array(bin_size)
    return bin_size / range_size

def random_bin_spawn_transform():
    x = random.uniform(-0.15, 0.15)
    y = 1.5
    z = -0.15
    position = np.array([x, y, z])

    z = random.random() * 0.02 - 0.01
    w = random.random() * 0.02 - 0.01
    norm = np.sqrt(z ** 2 + w ** 2)
    quat = math_util.Quaternion([w / norm, 0, 0, z / norm])
    # if random.random() > 0.5:
    #     print("<flip>")
    #     # flip the bin so it's upside down
    #     quat = quat * math_util.Quaternion([0, 0, 1, 0])
    # else:
    #     print("<no flip>")
    #print("<flip>")
    quat = quat * math_util.Quaternion([0, 0, 1, 0])

    return position, quat.vals


class BinStackingTask(BaseTask):
    def __init__(self, env_path, assets, pallet_scale):
        super().__init__("bin_stacking")
        self.assets = assets
        self.pallet_scale = pallet_scale

        self.env_path = "/World/Ur10Table"
        self.bins = []
        self.on_conveyor = None

    def _spawn_bin(self, rigid_bin):
        x, q = random_bin_spawn_transform()
        rigid_bin.set_world_pose(position=x, orientation=q)
        rigid_bin.set_linear_velocity(np.array([0, -0.30, 0]))
        rigid_bin.set_visibility(True)

    def post_reset(self) -> None:
        if len(self.bins) > 0:
            for rigid_bin in self.bins:
                self.scene.remove_object(rigid_bin.name)
            self.bins.clear()

        self.on_conveyor = None

    def pre_step(self, time_step_index, simulation_time) -> None:
        """ Spawn a new randomly oriented bin if the previous bin has been placed.
        """
        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            (x, y, z), _ = self.on_conveyor.get_world_pose()
            is_on_conveyor = y > 0.4 and -0.5 < x and x < 0.5
            if not is_on_conveyor:
                spawn_new = True

        if spawn_new:
            name = "bin_{}".format(len(self.bins))
            prim_path = self.env_path + "/bins/{}".format(name)
            add_reference_to_stage(usd_path=self.assets.box_usd, prim_path=prim_path)
            # add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)

            # scale bin from randomly chosen bin size
            bin_size = np.random.choice([0.2, 0.3, 0.4, 0.5], 2, True, p=[0.4, 0.3, 0.2, 0.1]) * self.pallet_scale
            bin_size -= 0.02
            bin_size = np.append(bin_size, 0.15 * self.pallet_scale)
            
            scale = get_scale(prim_path, bin_size)
            
            self.on_conveyor = self.scene.add(CustomCortexRigidPrim(
                name=name, prim_path=prim_path, scale=scale))
            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)


def main(agent, args):
    world = CortexWorld(physics_dt=0.02, rendering_dt=0.05)

    env_path = "/World/Ur10Table"
    
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(7.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-7.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    
    ur10_assets = Ur10Assets()
    add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)

    robot = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))

    obs = world.scene.add(
        VisualSphere(
            "/World/Ur10Table/Obstacles/NavigationDome",
            name="navigation_dome_obs",
            position=[-0.031, -0.018, -1.086],
            radius=1.1,
            visible=False,
        )
    )
    robot.register_obstacle(obs)

    az = np.array([1.0, 0.0, -0.3])
    ax = np.array([0.0, 1.0, 0.0])
    ay = np.cross(az, ax)
    R = math_util.pack_R(ax, ay, az)
    quat = math_util.matrix_to_quat(R)
    obs = world.scene.add(
        VisualCapsule(
            "/World/Ur10Table/Obstacles/NavigationBarrier",
            name="navigation_barrier_obs",
            position=[0.471, 0.276, -0.463 - 0.1],
            orientation=quat,
            radius=0.5,
            height=0.6,
            visible=False,
        )
    )
    robot.register_obstacle(obs)
    
    # Camera
    pallet_prim = world.scene.add(XFormPrim(prim_path=f"{env_path}/pallet", name="pallet"))
    pallet_pose = pallet_prim.get_world_pose()
    eye_position = pallet_pose[0]
    eye_position[0] += 0.9
    eye_position[2] += 3.5
    
    camera = Camera(
        prim_path="/World/camera",
        position=eye_position,
        frequency=10,
        resolution=(640, 480),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 105, 0]), degrees=True),
    )

    world.add_task(BinStackingTask(env_path, ur10_assets, args.pallet_scale))
    world.add_decider_network(behavior.make_decider_network(robot, agent, args))
    world.add_camera(camera)

    world.run(simulation_app, render=True, loop_fast=True, play_on_entry=True)
    simulation_app.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    ## Env ##
    parser.add_argument("--resolution", default=10, type=int)
    parser.add_argument("--max_levels", default=5, type=int)
    parser.add_argument("--pallet_scale", default=0.8, type=float)
    ## Agent ##
    parser.add_argument("--algorithm", default='DQN', type=str, help='[DQN, D-PPO, D-TAC]')
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--num_trials", default=25, type=int)
    ## Stack Condtions ##
    parser.add_argument("--use_bound_mask", action="store_false")
    parser.add_argument("--use_floor_mask", action="store_false")
    parser.add_argument("--use_projection", action="store_false")
    parser.add_argument("--use_coordnconv", action="store_false")
    args = parser.parse_args()

    if args.algorithm == "DQN":
        from agent.DQN import DQN_Agent as Agent
        model_path = os.path.join("results/models/DQN_%s.pth"%args.model_path)

    elif args.algorithm == "D-PPO":
        from agent.D_PPO import DiscretePPO_Agent as Agent
        args.show_q = False
        model_path = [
            os.path.join("results/models/DPPO_%s_critic.pth"%args.model_path),
            os.path.join("results/models/DPPO_%s_actor.pth"%args.model_path),
        ]

    elif args.algorithm == "D-TAC":
        from agent.D_TAC import DiscreteTAC_Agent as Agent
        model_path = [
            os.path.join("results/models/DTAC_%s_critic.pth"%args.model_path),
            os.path.join("results/models/DTAC_%s_actor.pth"%args.model_path),
        ]

    agent = Agent(False, model_path, args.use_coordnconv, args)
    main(agent, args)
