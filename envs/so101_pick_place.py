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
        # FINGER PAD GEOMS (for contact-based grasp detection)
        # ==================================================
        
        self.fixed_pad_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4"]
        ]
        
        self.moving_pad_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4"]
        ]

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
        
        self.bowl_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "bowl",
        )
        
        self.bowl_bottom_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "bowl_bottom",
        )

        # ==================================================
        # TABLE GEOMETRY (for lift detection)
        # ==================================================
        # Table body at z=0.35, table_top geom size z=0.02 -> table surface at z=0.37
        # Cube half-height = 0.02 -> cube center at 0.39 when on table
        self.table_surface_z = 0.37
        self.cube_half_height = 0.02
        self.cube_rest_z = self.table_surface_z + self.cube_half_height  # 0.39

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
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.12,
        ])

        self.max_episode_steps = 5000
        self.step_count = 0

        self.target = np.zeros(3)

        self.grasped = False
        self.was_lifted = False
        self.grasp_stage = 0  # 0=reach_above, 1=pitch_down, 2=descend, 3=close_grasp, 4=lift, 5=success

        self.viewer = None

        # Grasp sequence constants (derived from MuJoCo geometry)
        self.cube_rest_z = 0.39          # Cube center on table
        self.table_surface_z = 0.37      # Table top surface
        self.target_pitch_down = 1.57    # Wrist pitch for gripper pointing down
        self.target_roll_grasp = 0.0     # Wrist roll for grasp (0 = aligned)
        self.above_margin = 0.18        # 18cm above cube 
        self.lift_target_z = 0.46        # Lift cube to 7cm above table

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
        # Random cube (within robot reach: max 0.207 from base at 0,0)
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
        self.grasp_stage = 0
        self.grasp_contact_frames = 0
        self.release_contact_frames = 0

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
    # CONTACT-BASED GRASP DETECTION
    # ======================================================

    def _check_grasp(self):
        """
        Detect grasp using MuJoCo contacts between cube and finger pads.
        Requires contact on BOTH fixed and moving jaw pads.
        """
        has_fixed_contact = False
        has_moving_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if contact involves cube and fixed jaw pads
            if (geom1 == self.cube_geom_id and geom2 in self.fixed_pad_geom_ids) or \
               (geom2 == self.cube_geom_id and geom1 in self.fixed_pad_geom_ids):
                has_fixed_contact = True

            # Check if contact involves cube and moving jaw pads
            if (geom1 == self.cube_geom_id and geom2 in self.moving_pad_geom_ids) or \
               (geom2 == self.cube_geom_id and geom1 in self.moving_pad_geom_ids):
                has_moving_contact = True

        return has_fixed_contact and has_moving_contact

    # ======================================================
    # LIFT DETECTION
    # ======================================================

    def _check_lift(self, cube_pos):
        """
        Detect if cube has been lifted from table while grasped.
        Lift = cube center significantly above table resting height (0.39).
        Table surface at 0.37, cube half-height 0.02 -> resting center at 0.39.
        Lift threshold: cube center > 0.45 (6cm above table, well above noise).
        """
        return cube_pos[2] > (self.cube_rest_z + 0.06)

    # ======================================================
    # RELEASE DETECTION
    # ======================================================

    def _check_release(self, gripper_joint):
        """
        Detect intentional release using gripper state and contact loss.
        Release = gripper opened significantly AND grasp contacts lost.
        Uses hysteresis to avoid contact noise.
        """
        # Gripper significantly open (home is 0.25, fully open ~0.6+)
        gripper_open = gripper_joint > 0.5

        # Check if grasp contacts are lost
        has_fixed_contact = False
        has_moving_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if (geom1 == self.cube_geom_id and geom2 in self.fixed_pad_geom_ids) or \
               (geom2 == self.cube_geom_id and geom1 in self.fixed_pad_geom_ids):
                has_fixed_contact = True

            if (geom1 == self.cube_geom_id and geom2 in self.moving_pad_geom_ids) or \
               (geom2 == self.cube_geom_id and geom1 in self.moving_pad_geom_ids):
                has_moving_contact = True

        contacts_lost = not (has_fixed_contact and has_moving_contact)

        return gripper_open and contacts_lost

    # ======================================================
    # SUCCESS DETECTION (CUBE INSIDE BOWL)
    # ======================================================

    def _check_success(self, cube_pos):
        """
        Check if cube is actually inside the bowl using bowl's local frame.
        Bowl geometry (from scene.xml):
        - Bowl body has freejoint, position randomized
        - Bowl bottom at local pos (0, 0, -0.005), size (0.055, 0.055, 0.005)
        - Walls at y=±0.05, x=±0.05, extending up to z=+0.055 from bowl center
        - Interior: |x| < 0.05, |y| < 0.05, z between bottom and wall top
        """
        # Transform cube position to bowl local frame
        bowl_xpos = self.data.xpos[self.bowl_body_id]
        bowl_xmat = self.data.xmat[self.bowl_body_id].reshape(3, 3)
        
        # Cube position relative to bowl center in world frame
        rel_pos_world = cube_pos - bowl_xpos
        
        # Transform to bowl local frame (inverse rotation)
        rel_pos_local = bowl_xmat.T @ rel_pos_world
        
        # Bowl interior bounds
        # Walls at local x=±0.05, y=±0.05, height extends to ~0.055 above bowl center
        # Bowl bottom at z=-0.005 with thickness 0.005 -> interior floor at z=-0.01
        # But we use cube center, so allow z up to ~0.045 (wall height - cube half-height)
        
        inside_xy = (abs(rel_pos_local[0]) < 0.045 and abs(rel_pos_local[1]) < 0.045)
        inside_z = (rel_pos_local[2] < 0.045 and rel_pos_local[2] > -0.02)
        
        return inside_xy and inside_z

    # ======================================================
    # TABLE COLLISION
    # ======================================================

    def _check_table_collision(self):
        """
        Check if EE (Fixed_Jaw body) collides with table.
        Table top at z=0.37. Only penalize if EE goes below table surface.
        """
        ee_z = self.data.xpos[self.ee_body_id][2]
        return ee_z < self.table_surface_z - 0.01  # small margin

    # ======================================================
    # GRASP-AND-LIFT REWARD (NEW - REPLACES ALL STAGES)
    # ======================================================

    def _compute_reward(self, cube_pos, ee_pos, gripper_joint, distance_to_cube, distance_to_target, action):
        """
        6-stage grasp-and-lift sequence:
        0. REACH_ABOVE: EE over cube at cube_z + 7.5cm, gripper open
        1. PITCH_DOWN: Wrist pitch → 1.57, Wrist roll → 0 (gripper pointing down)
        2. DESCEND: Lower finger tips to cube height (0.39), center XY, jaw width ~0.055
        3. CLOSE_GRASP: Close gripper to joint 0.55, achieve dual-pad contact
        4. LIFT: Raise cube to 0.46 (7cm above table)
        5. SUCCESS: Cube lifted and held
        
        CONTINUOUS REWARDS (apply throughout episode):
        - EE centered over cube (XY alignment)
        - Wrist orientation maintained (pitch down, roll aligned)
        """
        reward = 0.0

        # Finger tip positions
        fixed_tip = self.data.geom_xpos[self.fixed_pad_geom_ids[0]]
        moving_tip = self.data.geom_xpos[self.moving_pad_geom_ids[0]]
        tip_midpoint = 0.5 * (fixed_tip + moving_tip)
        jaw_width = np.linalg.norm(fixed_tip - moving_tip)
        tip_xy_error = np.linalg.norm(cube_pos[:2] - tip_midpoint[:2])

        # Wrist joints
        wrist_pitch = self.data.qpos[self.model.jnt_qposadr[self.joint_ids[3]]]
        wrist_roll = self.data.qpos[self.model.jnt_qposadr[self.joint_ids[4]]]

        # Errors
        xy_error = np.linalg.norm(cube_pos[:2] - ee_pos[:2])
        cube_top_z = cube_pos[2] + self.cube_half_height
        z_error = tip_midpoint[2] - cube_top_z  # positive = tips above cube
        target_above_z = cube_pos[2] + self.above_margin

        # ============================================================
        # CONTINUOUS REWARDS (apply at ALL stages)
        # ============================================================
        
        # 1. ALWAYS center EE over cube (XY) - the most important behavior
        reward += -15.0 * xy_error
        
        # 2. ALWAYS maintain wrist pointing down (pitch) and aligned (roll)
        pitch_error = abs(wrist_pitch - self.target_pitch_down)
        roll_error = abs(wrist_roll - self.target_roll_grasp)
        
        

        # ============================================================
        # STAGE-SPECIFIC REWARDS
        # ============================================================

        # ===== STAGE 0: REACH ABOVE CUBE =====
        if self.grasp_stage == 0:
            # Z: target cube_z + 12cm above
            z_target_error = abs(ee_pos[2] - target_above_z)
            reward += -12.0 * z_target_error

            # Gripper open
            open_progress = np.clip(gripper_joint / 0.6, 0.0, 1.0)
            reward += 3.0 * open_progress

            # Advance when positioned above cube
            if xy_error < 0.05 and z_target_error < 0.02 and open_progress > 0.5:
                self.grasp_stage = 1
                reward += 20.0
                print("STAGE 0 COMPLETE -> MOVING TO STAGE 1")

        # ===== STAGE 1: PITCH WRIST DOWN =====
        elif self.grasp_stage == 1:
            
            reward += -5.0 * pitch_error
            reward += -3.0 * roll_error

            # Gripper stays open
            open_progress = np.clip(gripper_joint / 0.6, 0.0, 1.0)
            reward += 2.0 * open_progress

            # Advance when wrist pitched down
            if pitch_error < 0.15 and roll_error < 0.15:
                self.grasp_stage = 2
                reward += 20.0
                print("stage 1 completed")

        # ===== STAGE 2: DESCEND & ALIGN =====
        elif self.grasp_stage == 2:
            reward += -5.0 * pitch_error
            reward += -3.0 * roll_error
            # Lower tips to cube height (z_error → 0)
            reward += -2.0 * abs(z_error)
            reward += -20.0 * tip_xy_error

            # Jaw width for grasp
            reward += -18.0 * abs(jaw_width - 0.055)

            # Advance when tips at cube height and aligned
            if z_error < 0.015 and xy_error < 0.02 and abs(jaw_width - 0.055) < 0.015:
                self.grasp_stage = 3
                reward += 25.0

        # ===== STAGE 3: CLOSE GRIP & GRASP =====
        elif self.grasp_stage == 3:
            reward += -5.0 * pitch_error
            reward += -3.0 * roll_error
            # Close gripper: open=0.25 → grasp=0.55
            close_progress = np.clip((gripper_joint - 0.25) / 0.30, 0.0, 1.0)
            reward += 20.0 * close_progress

            # Contact bonus (detected via _check_grasp in step())
            if self.grasped:
                reward += 50.0  # Grasp achieved!
                self.grasp_stage = 4

        # ===== STAGE 4: LIFT CUBE =====
        elif self.grasp_stage == 4:
            # Lift cube to target height
            lift_progress = np.clip((cube_pos[2] - self.cube_rest_z) / (self.lift_target_z - self.cube_rest_z), 0.0, 1.0)
            reward += 30.0 * lift_progress

            # Keep gripper closed
            close_progress = np.clip((gripper_joint - 0.25) / 0.30, 0.0, 1.0)
            reward += 10.0 * close_progress

            # Penalize dropping
            if not self.grasped:
                reward -= 50.0
                self.grasp_stage = 0  # Reset

            # Advance when lifted
            if self.was_lifted:
                reward += 100.0  # Big lift bonus
                self.grasp_stage = 5

        # ===== STAGE 5: SUCCESS (HOLD LIFT) =====
        elif self.grasp_stage == 5:
            # Maintain lift
            reward += 20.0
            if not self.grasped:
                reward -= 50.0
                self.grasp_stage = 0

        # Table collision penalty
        if self._check_table_collision():
            reward -= 5.0

        # Action regularization
        reward -= 0.001 * np.sum(np.square(action))

        return reward

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

        gripper_joint = self.data.qpos[
            self.model.jnt_qposadr[self.joint_ids[5]]
        ]

        # --------------------------------------------------
        # Grasp detection (with contact hysteresis)
        # --------------------------------------------------

        grasp_detected = self._check_grasp()

        if not self.grasped and grasp_detected:
            self.grasp_contact_frames += 1
            if self.grasp_contact_frames >= 2:  # Require 2 frames of contact
                self.grasped = True
                self.grasp_contact_frames = 0
        else:
            self.grasp_contact_frames = 0

        # --------------------------------------------------
        # Lift detection
        # --------------------------------------------------

        if self.grasped and not self.was_lifted:
            if self._check_lift(cube_pos):
                self.was_lifted = True

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = self._compute_reward(
            cube_pos, ee_pos, gripper_joint,
            distance_to_cube, distance_to_target, action
        )

        # Table collision penalty (already in _compute_reward, but keep as safety)
        if self._check_table_collision():
            reward -= 5.0

        # Success: grasp_stage == 5 (lifted and held)
        success = (self.grasp_stage == 5 and self.was_lifted)

        terminated = success
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "success": bool(success),
            "grasped": bool(self.grasped),
            "lifted": bool(self.was_lifted),
            "grasp_stage": self.grasp_stage,
            "cube_height": float(cube_pos[2]),
            "grasp_contacts": bool(grasp_detected),
            "gripper_joint": float(gripper_joint),
            "ee_height": float(ee_pos[2]),
            "tip_midpoint_z": float(0.5 * (self.data.geom_xpos[self.fixed_pad_geom_ids[0]][2] + 
                                             self.data.geom_xpos[self.moving_pad_geom_ids[0]][2])),
            "wrist_pitch": float(self.data.qpos[self.model.jnt_qposadr[self.joint_ids[3]]]),
            "wrist_roll": float(self.data.qpos[self.model.jnt_qposadr[self.joint_ids[4]]]),
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