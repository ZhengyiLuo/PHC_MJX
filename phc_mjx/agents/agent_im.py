import math
import time
import os
import torch

os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import pickle
from collections import defaultdict
import glob
import os
import sys
import os.path as osp
from tqdm import tqdm
import wandb
import numpy as np
import multiprocessing

from smpl_sim.agents.agent_humanoid import AgentHumanoid
from smpl_sim.learning.memory import Memory
from smpl_sim.learning.policy_gaussian import PolicyGaussian
from smpl_sim.learning.critic import Value
from smpl_sim.learning.policy_mcp import PolicyMCP
from smpl_sim.learning.mlp import MLP
from smpl_sim.learning.logger_txt import create_logger
from smpl_sim.utils.flags import flags
from smpl_sim.learning.learning_utils import to_test, to_device, to_cpu, get_optimizer
from smpl_sim.envs.tasks import *
from phc_mjx.envs.humanoid_im import *
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite


class AgentIM(AgentHumanoid):
    
    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        termination_history = self.env.get_termination_history()
        state.update(termination_history)
        return state
    
    def set_full_state_weights(self, state):
        super().set_full_state_weights(state)
        if (not self.cfg.test) and  "termination_history" in  state and self.cfg.learning.resume_history:
            self.env.set_termination_history(state) 
            self.env.resample_motions()
        
    
    def pre_epoch(self):
        if (self.epoch > 1) and self.epoch % self.cfg.env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            self.env.resample_motions()
        return super().pre_epoch()
    
    def setup_env(self):
        self.env = eval(self.cfg.env.task)(self.cfg)

    def eval_policy(self, epoch=0, dump=False):
        res_dict_acc = {}
        self.env.start_eval(im_eval = True)
        to_test(*self.sample_modules) # Sending test modeuls to cpu!!!
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                num_threads = 64
                for run_idx in self.env.forward_motions(num_threads = num_threads):
                    queue = multiprocessing.Queue()
                    for i in range(num_threads-1):
                        worker_args = (i+1, queue)
                        worker = multiprocessing.Process(target=self.eval_single_thread, args=worker_args)
                        worker.start()
                    res_dict_acc.update(self.eval_single_thread(0, None))
                    
                    for _ in range(num_threads - 1):
                        pid, res_dict= queue.get()
                        res_dict_acc.update(res_dict)
                ####### Comopute Metrics 
                metrics_all, metrics_succ, failed_keys = defaultdict(list), defaultdict(list), []
                for motion_key, res_dict in res_dict_acc.items():
                    [metrics_all[k].append(v)  for k, v in res_dict.items()]
                    if res_dict['succ']:
                        [metrics_succ[k].append(v)  for k, v in res_dict.items()]
                    else:
                        failed_keys.append(motion_key)
                        
                metrics_all_print  = {m: np.mean(np.concatenate(v)) if m != 'succ' else np.mean(v) for m, v in metrics_all.items()}
                metrics_succ_print = {m: np.mean(np.concatenate(v)) for m, v in metrics_succ.items() if m != 'succ'}
                        
                if len(metrics_succ_print) == 0:
                    print("No success!!!")
                    metrics_succ_print = metrics_all_print
                    
                print("------------------------------------------")
                print(f"Success Rate: {metrics_all_print['succ']:.5f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
                print("Failed keys: ", len(failed_keys), failed_keys[:10])
                
                eval_info = {
                    "eval_success_rate": metrics_all_print['succ'],
                    "accel_dist": metrics_succ_print['accel_dist'], 
                    "vel_dist": metrics_succ_print['vel_dist'], 
                    "mpjpeg_all": metrics_all_print['mpjpe_g'],
                    "mpjpeg_succ": metrics_succ_print['mpjpe_g'],
                    "mpjpel_all": metrics_all_print['mpjpe_l'],
                    "mpjpel_succ": metrics_succ_print['mpjpe_l'],
                    "mpjpe_succ_pa": metrics_succ_print['mpjpe_pa'], 
                }
        self.env.end_eval({"failed_keys": failed_keys})
        
        return eval_info
    
    def eval_single_thread(self, pid, queue):
        res_dicts = {}
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                obs_dict, info = self.env.reset(options = {"motion_id": pid}) # ZL: not the most elegant way of setting the motion id.
                state = self.preprocess_obs(obs_dict)
                for t in range(10000):
                    self.env.recrod_eval_states()
                    actions = self.policy_net.select_action(torch.from_numpy(state).to(self.dtype), True)[0].numpy()
                    
                    next_obs, reward, terminated, truncated, info = self.env.step(self.preprocess_actions(actions))
                    next_state = self.preprocess_obs(next_obs)
                    done = terminated or truncated

                    if done:
                        res_dicts = self.env.dump_record_eval_states()
                        
                        assert(len(res_dicts) == 1)
                        for k, v in res_dicts.items():
                            res_dicts[k] = compute_metrics_lite(v['pred_jpos'][None, ], v['gt_jpos'][None, ], root_idx = 0, use_tqdm = False, concatenate = True)
                            res_dicts[k]['succ'] = not terminated
                        break 
                    state = next_state
                    
        if queue == None:
            return res_dicts
        else:
            queue.put((pid, res_dicts))
            
            
    def run_policy(self, epoch=0, dump=False):
        self.env.start_eval(im_eval = False)
        return super().run_policy(epoch, dump)