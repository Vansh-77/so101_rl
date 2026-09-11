import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class SO101PickPlaceEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        xml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "assets",
                "scene.xml",
            )
        )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================================================
        # JOINTS
        # ==================================================

        self.joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
            "Gripper",
        ]

        self.joint_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in self.joint_names
        ]

        # ==================================================
        # ACTUATORS
        # ==================================================

        self.actuator_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in self.joint_names
        ]

        # ==================================================
        # ROBOT BODIES
        # ==================================================

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Fixed_Jaw",
        )

        self.moving_jaw_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Moving_Jaw",
        )

        # ==================================================
        # CUBE
        # ==================================================

        self.cube_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )

        self.cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_freejoint",
        )

        self.cube_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube_geom",
        )

        # ==================================================
        # BOWL
        # ==================================================
        
        self.bowl_joint_id = mujoco.mj_name2id(
        self.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "bowl_freejoint",
        )
        
        # ==================================================
        # ACTION
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # ==================================================
        # OBSERVATION
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float32,
        )

        self.max_joint_delta = np.array([
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.10,
        ])

        self.max_episode_steps = 1000
        self.step_count = 0

        self.target = np.zeros(3)

        self.grasped = False
        self.was_lifted = False

        self.viewer = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # --------------------------------------------------
        # Robot home
        # --------------------------------------------------

        home = {
            "Shoulder_Rotation": 0.0,
            "Shoulder_Pitch": -1.57,
            "Elbow": 1.5,
            "Wrist_Pitch": 1.22,
            "Wrist_Roll": 0.0,
            "Gripper": 0.25,
        }

        for name, value in home.items():

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            qpos_addr = self.model.jnt_qposadr[
                joint_id
            ]

            self.data.qpos[qpos_addr] = value

        # --------------------------------------------------
        # Random cube
        # --------------------------------------------------
        MIN_SEPERATION = 0.16
        while True:
            cube_x = self.np_random.uniform(
                0.18,
                0.32,
            )

            cube_y = self.np_random.uniform(
                -0.18,
                0.18,
            )
            
            self.target = np.array([
            self.np_random.uniform(
                0.18,
                0.30,
            ),
            self.np_random.uniform(
                -0.20,
                0.20,
            ),
            0.41,
            ])
            
            separation = np.linalg.norm(
                np.array([cube_x, cube_y])
                - self.target[:2]
            )
            
            if separation >= MIN_SEPERATION:
                break
            
        cube_z = 0.39

        cube_qpos = self.model.jnt_qposadr[
            self.cube_joint_id
        ]

        self.data.qpos[
            cube_qpos:cube_qpos + 3
        ] = [
            cube_x,
            cube_y,
            cube_z,
        ]

        yaw = self.np_random.uniform(
            -np.pi,
            np.pi,
        )

        self.data.qpos[
            cube_qpos + 3:cube_qpos + 7
        ] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]

        # --------------------------------------------------
        # Random bowl position
        # --------------------------------------------------


        bowl_qpos = self.model.jnt_qposadr[
             self.bowl_joint_id
        ]
        self.data.qpos[bowl_qpos:bowl_qpos + 3] = self.target
        yaw = self.np_random.uniform(-np.pi, np.pi)

        self.data.qpos[bowl_qpos + 3:bowl_qpos + 7] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.step_count = 0
        self.grasped = False
        self.was_lifted = False

        return self._get_obs(), {}

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_obs(self):

        qpos = np.array([
            self.data.qpos[
                self.model.jnt_qposadr[j]
            ]
            for j in self.joint_ids
        ])

        qvel = np.array([
            self.data.qvel[
                self.model.jnt_dofadr[j]
            ]
            for j in self.joint_ids
        ])

        ee_pos = self.data.xpos[
            self.ee_body_id
        ].copy()

        cube_pos = self.data.xpos[
            self.cube_body_id
        ].copy()

        cube_to_ee = (
            cube_pos - ee_pos
        )

        cube_to_target = (
            self.target - cube_pos
        )

        return np.concatenate([
            qpos,
            qvel,
            ee_pos,
            cube_pos,
            self.target,
            cube_to_ee,
            cube_to_target,
        ]).astype(np.float32)

    # ======================================================
    # GRASP DETECTION
    # ======================================================

    def _check_grasp(self):

        cube_pos = self.data.xpos[
            self.cube_body_id
        ]

        fixed_pos = self.data.xpos[
            self.ee_body_id
        ]

        moving_pos = self.data.xpos[
            self.moving_jaw_body_id
        ]

        fixed_distance = np.linalg.norm(
            cube_pos - fixed_pos
        )

        moving_distance = np.linalg.norm(
            cube_pos - moving_pos
        )

        gripper_joint = self.data.qpos[
            self.model.jnt_qposadr[
                self.joint_ids[5]
            ]
        ]

        # Cube must be close to both jaws
        # and gripper must be reasonably closed.

        return (
            fixed_distance < 0.045
            and moving_distance < 0.045
            and gripper_joint < 0.45
        )

    # ======================================================
    # CHECK PLACEMENT
    # ======================================================

    def _check_success(self):

        cube_pos = self.data.xpos[
            self.cube_body_id
        ]

        horizontal_distance = np.linalg.norm(
            cube_pos[:2] -
            self.target[:2]
        )

        # Cube must be inside the bowl
        # and sitting approximately at the
        # bowl bottom.

        return (
            horizontal_distance < 0.045
            and cube_pos[2] < 0.45
            and not self.grasped
        )
    # ======================================================
    #  CHECK TABLE COLLISION
    # ======================================================
    
    def _check_table_collision(self):

        TABLE_Z_LIMIT = 0.40
        ee_z = self.data.xpos[self.ee_body_id][2]

        return ee_z < TABLE_Z_LIMIT

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # --------------------------------------------------
        # Joint position control
        # --------------------------------------------------

        for i, actuator_id in enumerate(
            self.actuator_ids
        ):

            joint_id = self.joint_ids[i]

            qpos_addr = self.model.jnt_qposadr[
                joint_id
            ]

            current = self.data.qpos[
                qpos_addr
            ]

            target = (
                current
                + action[i]
                * self.max_joint_delta[i]
            )

            target = np.clip(
                target,
                self.model.jnt_range[
                    joint_id,
                    0,
                ],
                self.model.jnt_range[
                    joint_id,
                    1,
                ],
            )

            self.data.ctrl[
                actuator_id
            ] = target

        # --------------------------------------------------
        # Simulation
        # --------------------------------------------------

        for _ in range(5):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        # --------------------------------------------------
        # State
        # --------------------------------------------------

        ee_pos = self.data.xpos[
            self.ee_body_id
        ]

        cube_pos = self.data.xpos[
            self.cube_body_id
        ]

        distance_to_cube = np.linalg.norm(
            ee_pos - cube_pos
        )

        distance_to_target = np.linalg.norm(
            cube_pos[:2] -
            self.target[:2]
        )
        reward = 0.0

        # --------------------------------------------------
        # Grasp
        # --------------------------------------------------

        if not self.grasped:

            if self._check_grasp():

                self.grasped = True
                reward +=20

        # --------------------------------------------------
        # Lift
        # --------------------------------------------------

        if self.grasped:

            if cube_pos[2] > 0.44:

                self.was_lifted = True

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        
        table_collision = self._check_table_collision()
        
        if table_collision:
            reward -= 5.0

        # Reach cube
        reward += -2.0 * distance_to_cube
        # grasp shaping
        
        fixed_pos = self.data.xpos[self.ee_body_id]
        moving_pos = self.data.xpos[self.moving_jaw_body_id]
        
        gripper_joint = self.data.qpos[
            self.model.jnt_qposadr[self.joint_ids[5]]
        ]
        
        jaw_midpoint = 0.5 * (
            fixed_pos + moving_pos
        )
        cube_to_midpoint = np.linalg.norm(
            cube_pos - jaw_midpoint
        )
        jaw_distance = np.linalg.norm(
            fixed_pos - moving_pos
        )
        desired_jaw_distance = 0.045

        if distance_to_cube < 0.10:

        # Get the cube into the middle of the gripper
            reward += -3.0 * cube_to_midpoint

            # Encourage the jaws to reach a graspable width
            reward += -2.0 * abs(
            jaw_distance - desired_jaw_distance
            )
             # Only encourage closing once the cube is
             # reasonably centered between the jaws.
             
            if cube_to_midpoint < 0.035:

                close_progress = np.clip(
                    (0.45 - gripper_joint) / 0.45,
                    0.0,
                    1.0,
                )
                reward += 2.0 * close_progress


        # Once grasped, reward transporting cube
        if self.grasped:

            reward += -3.0 * distance_to_target

        # Reward lifting
        if self.was_lifted:

            reward += 2.0

        # Successful placement
        success = self._check_success()

        if success and self.was_lifted:

            reward += 100.0

        # Action regularization
        reward -= 0.001 * np.sum(
            action ** 2
        )

        terminated = success and self.was_lifted

        truncated = (
            self.step_count
            >= self.max_episode_steps
        )

        info = {
            "success": bool(terminated),
            "grasped": bool(self.grasped),
            "lifted": bool(self.was_lifted),
            "cube_target_distance": float(
                distance_to_target
            ),
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

        self.viewer.sync()

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        if self.viewer is not None:

            self.viewer.close()

            self.viewer = None