# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, PhysxSchema

from omni.isaac.core.prims import XFormPrim


class CustomCortexRigidPrim(XFormPrim):
    def __init__(self, name, prim_path, position=None, orientation=None, scale=None):
        super().__init__(name=name, prim_path=prim_path, position=position, orientation=orientation, scale=scale)
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("Prim does not have the UsdPhysics.RigidBodyAPI schema.")
        self.rigid_api = UsdPhysics.RigidBodyAPI(self.prim)

    def enable_rigid_body_physics(self):
        self.rigid_api.GetRigidBodyEnabledAttr().Set(True)

    def disable_rigid_body_physics(self):
        self.rigid_api.GetRigidBodyEnabledAttr().Set(False)

    def get_linear_velocity(self):
        gf_velocity = self.rigid_api.GetVelocityAttr().Get()
        return np.array([gf_velocity[0], gf_velocity[1], gf_velocity[2]])

    def set_linear_velocity(self, velocity):
        gf_velocity = Gf.Vec3d(velocity[0], velocity[1], velocity[2])
        self.rigid_api.GetVelocityAttr().Set(gf_velocity)

    def get_angular_velocity(self):
        gf_ang_vel = self.rigid_api.GetAngularVelocityAttr().Get()
        return np.array([gf_ang_vel[0], gf_ang_vel[1], gf_ang_vel[2]])

    def set_angular_velocity(self, ang_vel):
        gf_ang_vel = Gf.Vec3d(ang_vel[0], ang_vel[1], ang_vel[2])
        self.rigid_api.GetAngularVelocityAttr().Set(gf_ang_vel)
