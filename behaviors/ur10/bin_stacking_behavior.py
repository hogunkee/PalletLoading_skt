# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import time

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.math import normalized
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.cortex.df import (
    DfNetwork,
    DfDecider,
    DfDecision,
    DfState,
    DfStateMachineDecider,
    DfStateSequence,
    DfTimedDeciderState,
    DfSetLockState,
    DfWaitState,
    DfWriteContextState,
)
from omni.isaac.cortex.dfb import make_go_home, DfLift, DfDiagnosticsMonitor
import omni.isaac.cortex.math_util as math_util
from omni.isaac.cortex.motion_commander import MotionCommand, ApproachParams, PosePq
from omni.isaac.cortex.obstacle_monitor_context import ObstacleMonitor, ObstacleMonitorContext

from behaviors.models.planners import NaivePlanner, NaiveRuleBasedPlanner, \
    RuleBasedPlanner, ModelBasedPlanner

from utils import get_bin_info, bbox_info, scaled_bbox_info

class BinState:
    def __init__(self, bin_obj):
        self.bin_obj = bin_obj
        self.bin_base = XFormPrim(self.bin_obj.prim_path)
        self.grasp_T = None
        self.is_grasp_reached = None
        self.is_attached = None


class NavigationObstacleMonitor(ObstacleMonitor):
    def __init__(self, context):
        obstacles = [
            context.world.scene.get_object("navigation_dome_obs"),
            context.world.scene.get_object("navigation_barrier_obs"),
            context.world.scene.get_object("navigation_flip_station_obs"),
        ]
        super().__init__(context, obstacles)

    def is_obstacle_required(self):
        target_p, _ = self.context.robot.arm.target_prim.get_world_pose()

        ref_p = np.array([0.6, 0.37, -0.99])
        eff_p = self.context.robot.arm.get_fk_p()

        ref_p[2] = 0.0
        eff_p[2] = 0.0
        target_p[2] = 0.0

        s_target = np.sign(np.cross(target_p, ref_p)[2])
        s_eff = np.sign(np.cross(eff_p, ref_p)[2])
        is_required = s_target * s_eff < 0.0
        return is_required


class BinStackingDiagnosticsMonitor(DfDiagnosticsMonitor):
    def print_diagnostics(self, context):
        # print("=========== logical state ==========")
        # if context.has_active_bin:
        #     print("active bin info:")
        #     print("- bin_obj.name: {}".format(context.active_bin.bin_obj.name))
        #     print("- bin_base: {}".format(context.active_bin.bin_base))
        #     print("- grasp_T:\n{}".format(context.active_bin.grasp_T))
        #     print("- is_grasp_reached: {}".format(context.active_bin.is_grasp_reached))
        #     print("- is_attached:  {}".format(context.active_bin.is_attached))
        # else:
        #     print("<no active bin>")

        # print("------------------------------------")
        pass

def get_bin_under(p, stacked_bins):
    x, y, z = p
    xy = np.array([x, y])

    for b in reversed(stacked_bins):
        (bin_x, bin_y, bin_z), _ = b.bin_obj.get_world_pose()
        bin_xy = np.array([bin_x, bin_y])
        if np.linalg.norm(bin_xy - xy) < 0.05:
            # We're searching in reverse. Return the first valid candidate.
            return b

    return None


class BinStackingContext(ObstacleMonitorContext):
    def __init__(self, robot, max_levels=3, resolution=10, model_path="./FCDQN_version5.pth"):
        super().__init__()
        self.robot = robot
        self.world = CortexWorld.instance()
        self.diagnostics_monitor = BinStackingDiagnosticsMonitor(print_dt=1.0)

        self.navigation_obs_monitor = NavigationObstacleMonitor(self)
        self.add_obstacle_monitors([self.navigation_obs_monitor])

        h = 0.135
        e = 0.0075
        x_shift = 0.05

        pallet = self.world.scene.get_object("pallet")
        self.pallet_pos, _ = pallet.get_world_pose()
        self.pallet_size = bbox_info(pallet.prim_path)
        
        # self.planner = RuleBasedPlanner(self.pallet_pos, self.pallet_size)# NaivePlanner()
        self.planner = ModelBasedPlanner(self.pallet_pos, self.pallet_size,
                                         resolution=10, max_levels=3,
                                         model_path=model_path)

        self.bins = []
        self.active_bin = None
        self.stacked_bins = []
        
        self.max_levels = max_levels
        self.resolution = resolution

        self.add_monitors(
            [
                BinStackingContext.monitor_bins,
                BinStackingContext.monitor_active_bin,
                BinStackingContext.monitor_active_bin_grasp_T,
                BinStackingContext.monitor_active_bin_grasp_reached,
                self.diagnostics_monitor.monitor,
            ]
        )

    def reset(self):
        super().reset()

        self.bins.clear()
        self.active_bin = None
        self.stacked_bins.clear()
        
        # reset planner stacked states
        self.planner.state = np.zeros((self.max_levels, self.resolution, self.resolution), dtype=np.uint8)
        self.planner.level_map = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        self.planner.poses = []
        self.planner.quats = []

    @property
    def stack_complete(self):
        # return len(self.stacked_bins) == len(self.stack_coordinates)
        # TODO: how should we define stack_complete?
        return self.stacked_volume / (1.0 * 1.0 * 0.15 * 3) > 0.9

    @property
    def stacked_volume(self):
        stacked_bin_observations, stacked_volume = \
            self.stacked_bin_observation()   
        return stacked_volume        

    @property
    def elapse_time(self):
        return time.time() - self.start_time

    @property
    def has_active_bin(self):
        return self.active_bin is not None

    def monitor_bins(self):
        if self.active_bin is None:
            self.conveyor_bin = None
            min_y = None

            # Check whether there's a new bin in the world.
            bin_obj = self.world.scene.get_object("bin_{}".format(len(self.bins)))
            if bin_obj is not None:
                self.bins.append(BinState(bin_obj))

            # Cycle through all bins and find the bin in the active region with smallest y value.
            for bin_state in self.bins:
                p, _ = bin_state.bin_obj.get_world_pose()

                # Check whether it's on the conveyor in the active region.
                x, y, z = p
                if 0.5 < y and y < 0.8 and -0.5 < x and x < 0.5:
                    if self.active_bin is None or y < min_y:
                        self.active_bin = bin_state
                        min_y = y

    def monitor_active_bin(self):
        if self.active_bin is not None:
            p, _ = self.active_bin.bin_obj.get_world_pose()
            if p[2] < -1.0:
                self.active_bin = None

    def monitor_active_bin_grasp_T(self):
        if self.active_bin is not None:
            bin_T = math_util.pq2T(*self.active_bin.bin_base.get_world_pose())
            bin_R, bin_p = math_util.unpack_T(bin_T)
            bin_ax, bin_ay, bin_az = math_util.unpack_R(bin_R)

            self.active_bin.is_rightside_up = False
            up_vec = np.array([0.0, 0.0, 1.0])
            if self.active_bin.is_attached:
                fk_R = self.robot.arm.get_fk_R()
                fk_x, _, _ = math_util.unpack_R(fk_R)
                up_vec = -fk_x

            bin_bbox = scaled_bbox_info(self.active_bin.bin_obj.prim_path, self.active_bin.bin_obj.get_local_scale())
            bin_half_depth = 0.5 * bin_bbox[2]

            target_ax = bin_az
            if bin_ax[1] < 0.0:
                # x axis is pointing toward the robot
                target_ay = -bin_ax
            else:
                target_ay = bin_ax
            target_az = np.cross(target_ax, target_ay)
            target_p = bin_p + bin_half_depth * (-target_ax)

            target_T = math_util.pack_Rp(math_util.pack_R(target_ax, target_ay, target_az), target_p)
            self.active_bin.grasp_T = target_T

    def monitor_active_bin_grasp_reached(self):
        if self.has_active_bin:
            fk_T = self.robot.arm.get_fk_T()
            self.active_bin.is_grasp_reached = math_util.transforms_are_close(
                self.active_bin.grasp_T, fk_T, p_thresh=0.005, R_thresh=0.01
            )
            # We can be looser with this proximity check.
            self.active_bin.is_attached = (
                math_util.transforms_are_close(self.active_bin.grasp_T, fk_T, p_thresh=0.1, R_thresh=1.0)
                and self.robot.suction_gripper.is_closed()
            )

    def mark_active_bin_as_complete(self):
        self.stacked_bins.append(self.active_bin)
        self.active_bin = None
    
    
    def stacked_bin_observation(self):
        observation = {}
        stacked_volume = 0.0
        for stacked_bin in self.stacked_bins:
            bin_obj = stacked_bin.bin_obj
            bin_info = get_bin_info(bin_obj)
            observation[bin_obj.name] = bin_info
            stacked_volume += bin_info["width"] * bin_info["height"] * bin_info["depth"]
        return observation, stacked_volume


class Move(DfState):
    def __init__(self, p_thresh, R_thresh):
        self.p_thresh = p_thresh
        self.R_thresh = R_thresh
        self.command = None

    def update_command(self, command):
        self.command = command

    def step(self):
        self.context.robot.arm.send(self.command)

        fk_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(
            self.command.target_pose.to_T(), fk_T, p_thresh=self.p_thresh, R_thresh=self.R_thresh
        ):
            return None
        return self


class MoveWithNavObs(Move):
    def enter(self):
        super().enter()
        self.context.navigation_obs_monitor.activate_autotoggle()

    def exit(self):
        super().exit()
        self.context.navigation_obs_monitor.deactivate_autotoggle()


class ReachToPick(MoveWithNavObs):
    """ Reach to pick the bin. The bin can be anywhere, including on the flip station. On entry, we
    activate the flip station obstacle monitor in case we're picking from the flip station. That
    obstacle monitor will prevent collision will the flip station en route.
    """

    def __init__(self):
        super().__init__(p_thresh=0.001, R_thresh=2.0)

    def enter(self):
        super().enter()

    def step(self):
        R, p = math_util.unpack_T(self.context.active_bin.grasp_T)
        ax, ay, az = math_util.unpack_R(R)

        posture_config = np.array([-1.2654234, -2.9708025, -2.219733, 0.6445836, 1.5186214, 0.30098662])

        approach_length = 0.1

        self.update_command(
            MotionCommand(
                target_pose=PosePq(p, math_util.matrix_to_quat(R)),
                approach_params=ApproachParams(direction=approach_length * ax, std_dev=0.005),
                posture_config=posture_config,
            )
        )

        return super().step()

    def exit(self):
        super().exit()

class ReachToPlace(MoveWithNavObs):
    def __init__(self):
        super().__init__(p_thresh=0.005, R_thresh=2.0)
        self.place_step = 0

    def enter(self):
        super().enter()

        bin_bbox = scaled_bbox_info(self.context.active_bin.bin_obj.prim_path, self.context.active_bin.bin_obj.get_local_scale())

        ############# --> need to fix. now planner consider rotation like code below.
        # if action_rot==0:
        #     by, bx, _ = next_block
        #     quat_ = [0.0, 0.0, 0.0, 1.0]

        # elif action_rot==1:
        #     bx, by, _ = next_block
        #     quat_ = [0.7071, 0.0, 0.0, 0.7071]
        # by, bx = math.ceil(by), math.ceil(bx)
        
        # also planner does not output z position. environment need to handle place z position offset.
        
        # box_level = np.max(self.level_map[max(min_y,0):max_y,max(min_x,0):max_x]) + 1
        # self.level_map[max(min_y,0):max_y,max(min_x,0):max_x] = box_level
        # (box_level-1)*self.box_height+0.01
        
        self.target_p, action_rot = self.context.planner.get_action(bin_bbox)
        self.bin_under = get_bin_under(self.target_p, self.context.stacked_bins)

        # TODO: calculate rotation matrix -> change to quaternion..?
        if action_rot == 0:
            target_ax = np.array([0.0, 0.0, -1.0])
            target_az = np.array([0.0, -1.0, 0.0])
            target_ay = np.cross(target_az, target_az)
        elif action_rot == 1:
            target_ax = np.array([0.0, 0.0, -1.0])
            target_az = np.array([-1.0, 0.0, 0.0])
            target_ay = np.cross(target_az, target_az)
        self.target_R = math_util.pack_R(target_ax, target_ay, target_az)
        #############

    def step(self):
        self.place_step += 1
        if self.bin_under is not None:
            bin_under_p, _ = self.bin_under.bin_obj.get_world_pose()
            bin_grasped_p, _ = self.context.active_bin.bin_obj.get_world_pose()
            xy_err = bin_under_p[:2] - bin_grasped_p[:2]
            if np.linalg.norm(xy_err) < 0.02:
                self.target_p[:2] += 0.1 * (bin_under_p[:2] - bin_grasped_p[:2])

        target_pose = PosePq(self.target_p, math_util.matrix_to_quat(self.target_R))

        approach_params = ApproachParams(direction=0.15 * np.array([0.0, 0.0, -1.0]), std_dev=0.005)
        posture_config = self.context.robot.default_config
        self.update_command(
            MotionCommand(target_pose=target_pose, approach_params=approach_params, posture_config=posture_config)
        )

        if self.place_step > 400:
            print("Planning takes too long!!! Resetting the environment")
            print("===============Evaluation Result===============")
            stacked_bin_observations, stacked_volume = \
                self.context.stacked_bin_observation()
            stacked_percentage = stacked_volume / (1.0 * 1.0 * 0.15 * 3) * 100
            print("Stacked: {} bins, {:.2f}% of the total volume".format(len(stacked_bin_observations), 
                                                    stacked_percentage))
            print("===============================================")
            self.place_step = 0
            self.context.world.reset(soft=False)

        return super().step()

    def exit(self):
        self.place_step = 0
        super().exit()

class CloseSuctionGripperWithRetries(DfState):
    def enter(self):
        pass

    def step(self):
        gripper = self.context.robot.suction_gripper
        gripper.close()
        if gripper.is_closed():
            return None
        return self


class CloseSuctionGripper(DfState):
    def enter(self):
        self.context.robot.suction_gripper.close()

    def step(self):
        return None


class OpenSuctionGripper(DfState):
    def enter(self):
        self.context.robot.suction_gripper.open()

    def step(self):
        return None


class DoNothing(DfState):
    def enter(self):
        self.context.robot.arm.clear()

    def step(self):
        #print(self.context.robot.arm.target_prim.get_world_pose())
        return self


class LiftAndTurn(Move):
    def __init__(self):
        super().__init__(p_thresh=0.05, R_thresh=0.1)

    def step(self):
        home_config = self.context.robot.default_config
        home_T = self.context.robot.arm.get_fk_T(config=home_config)

        p, q = math_util.T2pq(home_T)
        p += 0.5 * normalized(np.array([0.0, -0.5, -1.0]))
        self.target_pose = PosePq(p, q)
        self.update_command(MotionCommand(self.target_pose, posture_config=home_config))

        return super().step()


class PickBin(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPick(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    CloseSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.3), activity_duration=0.4),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class PlaceBin(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPlace(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.1), activity_duration=0.25),
                    DfWriteContextState(lambda ctx: ctx.mark_active_bin_as_complete()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class MoveToFlipStation(DfState):
    def __init__(self):
        self.target_pose = PosePq(
            np.array([0.7916634, 0.73902607, -0.02897218]), np.array([0.52239186, 0.6296602, -0.5042411, 0.27636158])
        )

        self.approach_params = ApproachParams(direction=0.4 * normalized(np.array([0.5, -0.3, -0.75])), std_dev=0.05)

        self.posture_config = np.array(
            [
                -2.1273114681243896,
                -3.004627227783203,
                -1.0576069355010986,
                -0.5193580389022827,
                -1.0809129476547241,
                2.0418107509613037,
            ]
        )

    def enter(self):
        motion_command = MotionCommand(
            target_pose=self.target_pose, approach_params=self.approach_params, posture_config=self.posture_config
        )
        self.context.robot.arm.send(motion_command)

    def step(self):
        fk_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(self.target_pose.to_T(), fk_T, p_thresh=0.005, R_thresh=2.0, verbose=False):
            return None
        return self


class ReleaseFlipStationBin(DfState):
    def __init__(self, duration):
        self.duration = duration

    def enter(self):
        self.entry_time = time.time()

        fk_T = self.context.robot.arm.get_fk_T()
        fk_R, fk_p = math_util.unpack_T(fk_T)
        ax, ay, az = math_util.unpack_R(fk_R)

        home_config = self.context.robot.default_config
        home_T = self.context.robot.arm.get_fk_T(config=home_config)
        home_R, home_p = math_util.unpack_T(home_T)

        v = normalized(np.array([-1.0, -0.3, 0.0]))
        toward_base_alpha = 0.2
        target_p = fk_p - 0.3 * ax + toward_base_alpha * v
        self.target_p = target_p
        self.ax = ax
        self.v = v

        target_ax = normalized(np.array([1.0, -0.0, 0.0]))
        target_ay = ay
        target_az = np.cross(target_ax, target_ay)
        target_R = math_util.pack_R(target_ax, target_ay, target_az)

        self.target_pose = PosePq(target_p, math_util.matrix_to_quat(target_R))
        motion_command = MotionCommand(
            target_pose=self.target_pose,
            approach_params=ApproachParams(direction=toward_base_alpha * v, std_dev=0.1),
            posture_config=self.context.robot.get_joint_positions().astype(float),
        )
        self.context.robot.arm.send(motion_command)

    def step(self):
        if time.time() - self.entry_time >= self.duration:
            return None
        return self


class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()

        self.add_child("pick_bin", PickBin())
        self.add_child("place_bin", PlaceBin())
        self.add_child("go_home", make_go_home())
        self.add_child("do_nothing", DfStateMachineDecider(DoNothing()))

    def decide(self):
        if self.context.stack_complete:
            print("Finished episode successfully!!!")
            print("===============Evaluation Result===============")         
            print("Stacked: {} bins, {:.2f}% of the total volume".format(len(self.context.stacked_bins), 
                                                    self.context.stacked_volume / (1.0 * 1.0 * 0.15 * 3) * 100))
            
            print("===============================================")
            return DfDecision("go_home")

        if self.context.has_active_bin:
            if not self.context.active_bin.is_attached:
                return DfDecision("pick_bin")
            else:
                return DfDecision("place_bin")
        else:
            return DfDecision("go_home")


def make_decider_network(robot, model_path="./FCDQN_version5.pth"):
    return DfNetwork(Dispatch(), context=BinStackingContext(robot, model_path=model_path))
