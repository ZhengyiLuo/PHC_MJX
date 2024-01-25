from typing import Any, Sequence
import numpy as np
from collections import OrderedDict
import torch
import mujoco
from collections import defaultdict
import copy

from smpl_sim.envs.humanoid_task import HumanoidTask
import smpl_sim.utils.np_transform_utils as npt_utils
from smpl_sim.utils.mujoco_utils import add_visual_capsule
from easydict import EasyDict
from smpl_sim.smpllib.motion_lib_smpl import MotionLibSMPL
from smpl_sim.smpllib.motion_lib_base import FixHeightMode
from smpl_sim.envs.humanoid_env import HumanoidEnv



class HumanoidIm(HumanoidTask):
    
    def __init__(self, cfg):
        self.num_traj_samples = cfg.env.get("num_traj_samples", 1) # paramter for number of future time steps
        self.reward_specs = cfg.env.reward_specs
        self.ref_motion_cache = EasyDict()
        self.global_offset = np.zeros([1, 3])
        self.termination_distance = cfg.env.termination_distance
        self.gender_betas = [np.zeros(17)] # current, all body shape is mean. 
        self.motion_start_idx = 0
        self.power_reward = cfg.env.power_reward
        self.power_coefficient = cfg.env.power_coefficient
        self.im_eval = cfg.im_eval
        self.test = cfg.test
        self.num_motion_max = 512
        self.im_obs_v = cfg.env.im_obs_v
        self.im_reward_v = cfg.env.im_reward_v
        super().__init__(cfg)
        self.setup_motionlib()
        
    def create_task_visualization(self):
        if self.viewer is not None: # this implies that headless == False
            for _ in range(len(self.track_bodies)):
                add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
                
        if self.renderer is not None:
            for _ in range(len(self.track_bodies)):
                add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
    
    def draw_task(self):
        def draw_obj(scene):
            time = (self.cur_t) * self.dt + self._motion_start_times + self._motion_start_times_offset # Reset is also called after the progress_buf is updated. 
            ref_dict = self.get_state_from_motionlib_cache(self._sampled_motion_ids, time, self.global_offset)
            ref_pos_subset = ref_dict.xpos[..., self.track_bodies_id, :]
            
            for i in range(len(self.track_bodies)):
                scene.geoms[i].pos = ref_pos_subset[0, i]

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)
        
    def setup_humanoid_properties(self):
        super().setup_humanoid_properties()
        self.full_track_bodies = self.body_names_orig
        self.track_bodies = self.cfg.env.get("trackBodies", self.full_track_bodies)
        self.reset_bodies = self.cfg.env.get("resetBodies", self.track_bodies)
        
        self.track_bodies_id = [self.body_names_orig.index(j) for j in self.track_bodies]
        self.reset_bodies_id = [self.body_names_orig.index(j) for j in self.reset_bodies]
        
    def setup_motionlib(self):
        self.motion_lib_cfg = EasyDict({
            "motion_file": self.cfg.env.motion_file,
            "device": torch.device("cpu"),
            "fix_height": FixHeightMode.full_fix,
            "min_length": -1,
            "max_length": -1,
            "multi_thread": True if self.cfg.num_threads > 1 else False,
            "smpl_type": "smpl",
            "randomrize_heading": not self.test,
        })
        self.motion_lib = MotionLibSMPL(self.motion_lib_cfg)
        if self.test:
            self.motion_lib.load_motions(self.motion_lib_cfg, shape_params = self.gender_betas, random_sample = False)
        else:
            self.motion_lib.load_motions(self.motion_lib_cfg, shape_params = self.gender_betas * min(self.num_motion_max, self.motion_lib.num_all_motions()), random_sample = True)
        
        self._sampled_motion_ids = np.array([0])
        self._motion_start_times = np.zeros(1)
        self._motion_start_times_offset = np.zeros(1)
        return
    
    def resample_motions(self):
        self.motion_lib.load_motions(self.motion_lib_cfg, shape_params = self.gender_betas * min(self.num_motion_max, self.motion_lib.num_all_motions()), random_sample = True)
    
    def forward_motions(self, num_threads = 32):
        for motion_start_idx in range(0, self.motion_lib.num_all_motions(), num_threads):
            self.motion_start_idx = motion_start_idx
            self.motion_lib.load_motions(self.motion_lib_cfg, shape_params = self.gender_betas * num_threads, random_sample = False, silent=False, start_idx = self.motion_start_idx)
            yield motion_start_idx
            
    def next_motions(self):
        self.motion_start_idx += 1
        self.motion_lib.load_motions(self.motion_lib_cfg, shape_params = self.gender_betas * self.cfg.num_threads, random_sample = False, silent=False, start_idx = self.motion_start_idx)
        
    def init_humanoid(self):
        if self.state_init == HumanoidEnv.StateInit.Default:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                self.mj_data.qpos[:] = 0
                self.mj_data.qvel[:] = 0
                self.mj_data.qpos[2] = 0.94
                self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])
        elif self.state_init == HumanoidEnv.StateInit.Fall:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                self.mj_data.qpos[:] = 0
                self.mj_data.qvel[:] = 0
                self.mj_data.qpos[2] = 0.3
                self.mj_data.qpos[3:7] = np.array([1, 0, 0, 0])
                mujoco.mj_forward(self.mj_model, self.mj_data)
                for _ in range(3):
                    # on purpose this is always done in torque space
                    action = (self.np_random.random(self.get_action_size()) - 0.5 ) * 1
                    for _ in range(self.control_freq_inv):
                        torque = self.compute_torque(action)
                        self.mj_data.ctrl[:] = torque
                        mujoco.mj_step(self.mj_model, self.mj_data)
        elif self.state_init == HumanoidEnv.StateInit.MoCap:
            motion_return = self.get_state_from_motionlib_cache(self._sampled_motion_ids, self._motion_start_times, self.global_offset)
            self.mj_data.qpos = motion_return.qpos[0]
            self.mj_data.qvel = motion_return.qvel[0]
    
    def get_task_obs_size(self):
        if self.im_obs_v == 1:
            obs_size = len(self.track_bodies) * self.num_traj_samples * (3 + 6 + 3 + 6) # 3 for position, 6 for rotation, 3 for ref position, 6 for angular ref rotation
            obs_size += (1 + 1 + len(self.track_bodies) - 1) * 3 # root velocity, root angular velocity, dof velocity
        elif self.im_obs_v == 2:
            assert(self.cfg.robot.create_vel_sensors)
            obs_size = len(self.track_bodies) * self.num_traj_samples * (3 + 6 + 3 + 3 + 3 + 6) # 3 for position, 6 for rotation, 3 for linear vel, 3 for angular vel, 3 for ref position, 6 for ref rotation
    
        return obs_size 
    
    def compute_reset(self):
        terminated, truncated = False, False
        time = (self.cur_t) * self.dt + self._motion_start_times + self._motion_start_times_offset # Reset is also called after the progress_buf is updated. 
        ref_dict = self.get_state_from_motionlib_cache(self._sampled_motion_ids, time, self.global_offset)
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]
        
        body_pos_subset = body_pos[..., self.reset_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., self.reset_bodies_id, :]
        terminated = compute_humanoid_im_reset(body_pos_subset, ref_pos_subset, termination_distance=self.termination_distance, use_mean=self.im_eval)[0]
        truncated = (time > self.motion_lib.get_motion_length(self._sampled_motion_ids))[0]
        return terminated, truncated
    
    
    def update_task(self):
        return

    def reset_task(self, options = None):
        if self.test:
            if self.im_eval:
                self._sampled_motion_ids[:] = options['motion_id']
                self._motion_start_times[:] = 0
            else:
                self._sampled_motion_ids[:] = 0
                self._motion_start_times[:] = 0
            # self._motion_start_times[:] = self.motion_lib.sample_time(self._sampled_motion_ids)
        else:
            self._sampled_motion_ids[:] = self.motion_lib.sample_motions()
            self._motion_start_times[:] = self.motion_lib.sample_time(self._sampled_motion_ids)
        return
    
    def get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  np.abs(self.ref_motion_cache['motion_ids'] - motion_ids).sum() + np.abs(self.ref_motion_cache['motion_times'] - motion_times).sum() + np.abs(self.ref_motion_cache['offset'] - offset).sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.copy()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.copy()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.copy() if not offset is None else None
        else:
            return self.ref_motion_cache
        
        motion_res = self.motion_lib.get_motion_state_intervaled(motion_ids.copy(), motion_times.copy(), offset=offset)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache
    
    def compute_task_obs(self):
        # mujoco.mj_kinematics(self.mj_model, self.mj_data)  # Do not need to run this since compute_proprioception happens before this
        motion_times = (self.cur_t + 1) * self.dt + self._motion_start_times + self._motion_start_times_offset  # Next frame, so +1
        ref_dict = self.get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self.global_offset)
        
        qpos = self.get_qpos()[None,]  # TODO why qpos is not used in the proprioception?
        qvel = self.get_qvel()[None,]
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]
        
        body_pos_subset = body_pos[..., self.track_bodies_id, :]
        body_rot_subset = body_rot[..., self.track_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., self.track_bodies_id, :]
        ref_rot_subset = ref_dict.xquat[..., self.track_bodies_id, :]
        
        if self.im_obs_v == 1:
            ref_qvel = ref_dict.qvel
            task_obs = compute_imitation_observations_v1(qpos, qvel, body_pos_subset, body_rot_subset,  ref_pos_subset, ref_rot_subset, ref_qvel, self.num_traj_samples, upright=self.upright_start, humanoid_type = self.humanoid_type)
        elif self.im_obs_v == 2:
            body_vel = self.get_body_linear_vel()[None,]
            body_ang_vel = self.get_body_angular_vel()[None,]
            ref_body_vel_subset = ref_dict.body_vel[..., self.track_bodies_id, :]
            ref_body_ang_vel_subset = ref_dict.body_ang_vel[..., self.track_bodies_id, :]
            
            task_obs = compute_imitation_observations_v2(qpos, qvel, body_pos_subset, body_rot_subset, body_vel, body_ang_vel,  ref_pos_subset, ref_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, self.num_traj_samples, upright=self.upright_start, humanoid_type = self.humanoid_type)
            
        
        return np.concatenate([v.ravel() for v in task_obs.values()], axis=0, dtype=self.dtype)

    def recrod_eval_states(self):
        motion_times = (self.cur_t) * self.dt + self._motion_start_times + self._motion_start_times_offset  # Next frame, so +1
        motion_res = self.motion_lib.get_motion_state_intervaled(self._sampled_motion_ids, motion_times.copy(), offset=self.global_offset)
        
        self.state_record['pred_jpos'].append(self.get_body_xpos())
        self.state_record['gt_jpos'].append(motion_res.xpos.squeeze())
    
    def dump_record_eval_states(self):
        return_dict =  {
            self.motion_lib.curr_motion_keys[self._sampled_motion_ids][0]: {k:np.array(v) for k, v in self.state_record.items()}
        }
        self.state_record = defaultdict(list)
        return return_dict

    def compute_reward(self, actions):
        
        motion_times = self.cur_t  * self.dt + self._motion_start_times + self._motion_start_times_offset  # Next frame, so +1
        ref_dict = self.get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self.global_offset)

        ###### Debugging
        # self.mj_data.qpos[:] = ref_dict.qpos[0]
        # self.mj_data.qvel[:] = ref_dict.qvel[0]
        # mujoco.mj_forward(self.mj_model, self.mj_data)
        #######
        
        qpos = self.get_qpos()[None,]  # TODO why qpos is not used in the proprioception?
        qvel = self.get_qvel()[None,]
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]


        body_pos_subset = body_pos[..., self.track_bodies_id, :]
        body_rot_subset = body_rot[..., self.track_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., self.track_bodies_id, :]
        ref_rot_subset = ref_dict.xquat[..., self.track_bodies_id, :]
        
        if self.im_reward_v == 1:
            ref_qvel = ref_dict.qvel
            reward, reward_raw = compute_imitation_reward_v1(qpos, qvel, body_pos_subset, body_rot_subset,  ref_pos_subset, ref_rot_subset, ref_qvel, self.reward_specs)
        elif self.im_reward_v == 2:
            body_vel = self.get_body_linear_vel()[None,]
            body_ang_vel = self.get_body_angular_vel()[None,]
            ref_body_vel_subset = ref_dict.body_vel[..., self.track_bodies_id, :]
            ref_body_ang_vel_subset = ref_dict.body_ang_vel[..., self.track_bodies_id, :]
            
            reward, reward_raw = compute_imitation_reward_v2(qpos, qvel, body_pos_subset, body_rot_subset, body_vel, body_ang_vel,  ref_pos_subset, ref_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, self.reward_specs)

        
        if self.power_reward:
            power_reward = self.compute_power_reward()
            reward = reward + power_reward
            reward_raw["power_reward"] = power_reward
        
        self.reward_info = reward_raw
        return reward[0]
    
    def compute_power_reward(self):
        return -self.power_coefficient * np.array(self.curr_power_usage).mean(axis = 0).sum()
    
    def start_eval(self, im_eval = True):
        self.motion_lib_cfg.randomrize_heading = False
        self.im_eval = im_eval
        self.test = True
        
        self._temp_termination_distance = self.termination_distance
        self.termination_distance = 0.5
    
    def end_eval(self, results):
        self.motion_lib_cfg.randomrize_heading = True
        self.im_eval = False
        self.test = False
        self.termination_distance = self._temp_termination_distance
        self.motion_lib.update_soft_sampling_weight(results['failed_keys'])
        self.resample_motions()
        
    
    def get_termination_history(self):
        return self.motion_lib.get_termination_history()
    
    def set_termination_history(self, state):
        self.motion_lib.set_termination_history(state)
    
    
        
    def key_callback(self, keycode):
        super().key_callback(keycode)
        if chr(keycode) == "T":
            self.next_motions()
            self.reset()

def compute_imitation_observations_v1(qpos, qvel, body_pos, body_rot, ref_body_pos, ref_body_rot, ref_qvel, time_steps, upright, humanoid_type = "smpl"):
    # We do not use any dof in observation.
    # the design of the ref_qvel isn't good. 
    obs = OrderedDict()
    B, J, _ = body_pos.shape
    root_rot = qpos[:, 3:7]
    root_pos = qpos[:, :3]

    if not upright:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)
    heading_rot = npt_utils.calc_heading_quat(root_rot)
    
    heading_inv_rot_expand = np.tile(heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis = 1), (time_steps, 1, 1))
    heading_rot_expand = np.tile(heading_rot[..., None, :, :].repeat(body_pos.shape[1], axis = 1), (time_steps, 1, 1))
    

    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(B, 1, J, 3)
    diff_global_body_rot = npt_utils.quat_mul(ref_body_rot.reshape(B, time_steps, J, 4), np.tile(npt_utils.quat_conjugate(body_rot), (time_steps, 1, 1)).reshape(B, time_steps, J, 4))
    
    
    diff_local_body_pos_flat = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))
    diff_local_body_rot_flat = npt_utils.quat_mul(npt_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_rot.reshape(-1, 4)), heading_rot_expand.reshape(-1, 4))  # Need to be change of basis

    obs['diff_local_body_pos'] = diff_local_body_pos_flat  # 1 * J * 3 
    obs['diff_local_body_rot'] = npt_utils.quat_to_tan_norm(diff_local_body_rot_flat)  #  1 * J * 6

    ##### Velocities
    root_velp = qvel[:, None, 0:3]
    root_velr = qvel[:, None, 3:6]
    body_vel = qvel[:, 6:]
    
    ref_velp = ref_qvel[:, None, 0:3]
    ref_velr = ref_qvel[:, None, 3:6]
    ref_body_vel = ref_qvel[:, 6:]
    
    diff_root_vel = root_velp.reshape(B, time_steps, 1, 3) - ref_velp.reshape(B, 1, 1, 3)
    diff_root_ang_vel = root_velr.reshape(B, time_steps, 1, 3) - ref_velr.reshape(B, 1, 1, 3)

    obs['diff_local_root_vel'] = npt_utils.quat_rotate(heading_inv_rot.reshape(-1, 4), diff_root_vel.reshape(-1, 3))
    obs['diff_local_root_ang_vel'] = npt_utils.quat_rotate(heading_inv_rot.reshape(-1, 4), diff_root_ang_vel.reshape(-1, 3))
    obs['diff_dof_vel'] =  ref_body_vel.reshape(B, time_steps, J - 1, 3) - body_vel.reshape(B, 1, J - 1, 3)
    
    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(B, 1, 1, 3)  # preserves the body position
    obs['local_ref_body_pos'] = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)).reshape(B, -1)  #  1 * J * 3

    local_ref_body_rot = npt_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), ref_body_rot.reshape(-1, 4))
    obs['local_ref_body_rot'] = npt_utils.quat_to_tan_norm(local_ref_body_rot) #  1 * J * 6
    
    return  obs


def compute_imitation_observations_v2(qpos, qvel, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright, humanoid_type = "smpl"):
    # We do not use any dof in observation.
    # the design of the ref_qvel isn't good. 
    obs = OrderedDict()
    B, J, _ = body_pos.shape
    root_rot = qpos[:, 3:7]
    root_pos = qpos[:, :3]

    if not upright:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)
    heading_rot = npt_utils.calc_heading_quat(root_rot)
    
    heading_inv_rot_expand = np.tile(heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis = 1), (time_steps, 1, 1))
    heading_rot_expand = np.tile(heading_rot[..., None, :, :].repeat(body_pos.shape[1], axis = 1), (time_steps, 1, 1))
    

    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(B, 1, J, 3)
    diff_global_body_rot = npt_utils.quat_mul(ref_body_rot.reshape(B, time_steps, J, 4), np.tile(npt_utils.quat_conjugate(body_rot), (time_steps, 1, 1)).reshape(B, time_steps, J, 4))
    
    
    diff_local_body_pos_flat = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3))
    diff_local_body_rot_flat = npt_utils.quat_mul(npt_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), diff_global_body_rot.reshape(-1, 4)), heading_rot_expand.reshape(-1, 4))  # Need to be change of basis

    obs['diff_local_body_pos'] = diff_local_body_pos_flat  # 1 * J * 3 
    obs['diff_local_body_rot'] = npt_utils.quat_to_tan_norm(diff_local_body_rot_flat)  #  1 * J * 6

    ##### Velocities
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(B, 1, J, 3)
    obs['diff_local_vel'] = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3))

    diff_global_ang_vel = ref_body_ang_vel.reshape(B, time_steps, J, 3) - body_ang_vel.reshape(B, 1, J, 3)
    obs['diff_local_ang_vel'] = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), diff_global_ang_vel.reshape(-1, 3))
    
    
    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(B, 1, 1, 3)  # preserves the body position
    obs['local_ref_body_pos'] = npt_utils.quat_rotate(heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)).reshape(B, -1)  #  1 * J * 3

    local_ref_body_rot = npt_utils.quat_mul(heading_inv_rot_expand.reshape(-1, 4), ref_body_rot.reshape(-1, 4))
    obs['local_ref_body_rot'] = npt_utils.quat_to_tan_norm(local_ref_body_rot) #  1 * J * 6
    
    return  obs

def compute_imitation_reward_v1(qpos, qvel, body_pos, body_rot, ref_body_pos, ref_body_rot, ref_qvel, rwd_specs):
    k_pos, k_rot, k_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"]
    w_pos, w_rot, w_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(axis=-1).mean(axis=-1)
    r_body_pos = np.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = npt_utils.quat_mul(ref_body_rot, npt_utils.quat_conjugate(body_rot))
    diff_global_body_angle = npt_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(axis=-1)
    r_body_rot = np.exp(-k_rot * diff_global_body_angle_dist)

    
    ##### Velocities
    root_velp = qvel[:, None, 0:3]
    root_velr = qvel[:, None, 3:6]
    body_vel = qvel[:, 6:]
    
    ref_velp = ref_qvel[:, None, 0:3]
    ref_velr = ref_qvel[:, None, 3:6]
    ref_body_vel = ref_qvel[:, 6:]
    
    # root linear velocity reward
    diff_root_vel = ref_velp - root_velp
    diff_global_vel_dist = (diff_root_vel**2).mean(axis=-1).mean(axis=-1)
    r_root_vel = np.exp(-k_vel * diff_global_vel_dist)
    

    # root angular velocity reward
    diff_global_ang_vel = ref_velr - root_velr
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(axis=-1).mean(axis=-1)
    r_root_ang_vel = np.exp(-k_vel * diff_global_ang_vel_dist)
    
    # dof vel
    diff_dof_vel = ref_body_vel - body_vel
    diff_dof_vel_dist = (diff_dof_vel**2).mean(axis=-1).mean(axis=-1)
    r_dof_vel = np.exp(-k_vel * diff_dof_vel_dist)
    
    r_vel = (r_root_vel + r_root_ang_vel + r_dof_vel)/3
    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel
    reward_raw = {"r_body_pos":r_body_pos, "r_body_rot":r_body_rot,  "r_root_vel":r_root_vel,  "r_root_ang_vel":r_root_ang_vel, "r_dof_vel":r_dof_vel}
    
    return reward, reward_raw

def compute_imitation_reward_v2(qpos, qvel, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    k_pos, k_rot, k_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"]
    w_pos, w_rot, w_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(axis=-1).mean(axis=-1)
    r_body_pos = np.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = npt_utils.quat_mul(ref_body_rot, npt_utils.quat_conjugate(body_rot))
    diff_global_body_angle = npt_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(axis=-1)
    r_body_rot = np.exp(-k_rot * diff_global_body_angle_dist)

    
    ##### Velocities
    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(axis=-1).mean(axis=-1)
    r_vel = np.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(axis=-1).mean(axis=-1)
    r_ang_vel = np.exp(-k_vel * diff_global_ang_vel_dist)
    
    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel
    reward_raw = {"r_body_pos":r_body_pos, "r_body_rot":r_body_rot,  "r_vel":r_vel,  "r_ang_vel":r_ang_vel}
    
    return reward, reward_raw




def compute_humanoid_im_reset(rigid_body_pos, ref_body_pos, termination_distance, use_mean):
    if use_mean:
        has_fallen = np.any(np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1).mean(axis=-1, keepdims=True) > termination_distance, axis=-1)  # using average, same as UHC"s termination condition
    else:
        has_fallen = np.any(np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1) > termination_distance, axis=-1)  # using max

    return has_fallen