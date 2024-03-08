"""  All classes for physics (forward dynamics), update the states of the drone in the simulator PyBullet
Basic codes:
    https://github.com/SvenGronauer/phoenix-drone-simulation/tree/master 
    https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/envs/BaseAviary.py#L717 

Hanyang Hu, Xubo Lyu, Xilun Zhang
SFU Mars Lab, CMU SafeAI Lab
2024.03.01
"""

import abc
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client
from typing import Tuple



class BasePhysics(abc.ABC):
    """Parent class."""

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,
            gravity: float = 9.81,
            number_solver_iterations: int = 5,
            use_ground_effect: bool = False
    ):
        self.drone = drone
        self.bc = bc
        self.time_step = time_step
        self.G = gravity
        self.number_solver_iterations = number_solver_iterations
        self.use_ground_effect = use_ground_effect

    def calculate_ground_effect(
            self,
            motor_forces: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        r"""Implementation of a ground effect model.

        Taken from:
        https://github.com/utiasDSL/gym-pybullet-drones/blob/a133c163e533ef1f5c55d7c1c631653e17f3bd79/gym_pybullet_drones/envs/BaseAviary.py#L709
        """
        # Kinematic info of all links (propellers and center of mass)
        link_states = self.bc.getLinkStates(
            self.drone.body_unique_id,
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        gec, r = self.drone.GND_EFF_COEFF, self.drone.PROP_RADIUS
        roll, pitch = self.drone.rpy[0:2]

        prop_z = np.array([
            link_states[0][0][2],  # z-position of link 0
            link_states[1][0][2],  # z-position of link 1
            link_states[2][0][2],  # z-position of link 2
            link_states[3][0][2]  # z-position of link 3
        ])
        prop_z = np.clip(prop_z, self.drone.GND_EFF_H_CLIP, np.inf)
        # Simple, per-propeller ground effects
        gnd_effects = motor_forces * gec * (r/(4 * prop_z))**2
        if np.abs(roll) < np.pi/2 and np.abs(pitch) < np.pi/2:
            return True, gnd_effects
        else:
            return False, np.zeros_like(gnd_effects)

    def set_parameters(
            self,
            time_step: float,
            number_solver_iterations: int,
            # **kwargs
    ):
        self.time_step = time_step
        self.number_solver_iterations = number_solver_iterations

    @abc.abstractmethod
    def step_forward(self,
                     action: np.ndarray,
                     *args,
                     **kwargs
                     ) -> None:
        r"""Steps the physics once forward."""
        raise NotImplementedError


class PyBulletPhysics(BasePhysics):
    
    def set_parameters(self, *args, **kwargs):
        super(PyBulletPhysics, self).set_parameters(*args, **kwargs)
        # Update PyBullet Physics
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1
        )

    def step_forward(self, action, *args, **kwargs):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        action
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        motor_forces, z_torque = self.drone.apply_action(action)

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.drone.apply_motor_forces(motor_forces)
        self.drone.apply_z_torque(z_torque)

        # === add drag effect
        quat = self.drone.quaternion
        vel = self.drone.xyz_dot
        base_rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Simple draft model applied to the base/center of mass
        rpm = self.drone.x**2 * 25000
        drag_factors = -1 * self.drone.DRAG_COEFF * np.sum(2*np.pi*rpm/60)
        drag = np.dot(base_rot, drag_factors*np.array(vel))
        # print(f'Drag: {drag}')
        self.drone.apply_force(force=drag)

        # === Ground Effect
        apply_ground_eff, ge_forces = self.calculate_ground_effect(motor_forces)
        if apply_ground_eff and self.use_ground_effect:
            self.drone.apply_motor_forces(forces=ge_forces)

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()


class PybulletPhysicsWithDistb(PyBulletPhysics):
    def set_parameters(self, *args, **kwargs):
        super(PybulletPhysicsWithDistb, self).set_parameters(*args, **kwargs)
        # Update PyBullet Physics
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1
        )

    def step_forward(self, action, distb, *args, **kwargs):
        """ PyBullet physics with adversary effect included implementation.

        Parameters
        ----------
        action: pure action to to be applied to the drone
        distb: pure adversarial disturbance to be applied to the drone
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        motor_forces, z_torque = self.drone.apply_action(action)

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.drone.apply_motor_forces(motor_forces)
        self.drone.apply_z_torque(z_torque)

        # === XL: add adversary effect
        self.drone.apply_x_torque(distb[0])
        self.drone.apply_y_torque(distb[1])

        # === add drag effect
        quat = self.drone.quaternion
        vel = self.drone.xyz_dot
        base_rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Simple draft model applied to the base/center of mass
        rpm = self.drone.x**2 * 25000
        drag_factors = -1 * self.drone.DRAG_COEFF * np.sum(2*np.pi*rpm/60)
        drag = np.dot(base_rot, drag_factors*np.array(vel))
        # print(f'Drag: {drag}')
        self.drone.apply_force(force=drag)

        # === Ground Effect
        apply_ground_eff, ge_forces = self.calculate_ground_effect(motor_forces)
        if apply_ground_eff and self.use_ground_effect:
            self.drone.apply_motor_forces(forces=ge_forces)

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()
