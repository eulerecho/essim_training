import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from math import sqrt
import time

class EssimEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'essim.xml', 2)
        # test initial condition #self.init_qpos = np.array([0.012566, -0.0605091, -7.64849e-05, 0.999888, -0.000885256, -0.000537436, -0.0149548, 0.39768, 0.288012, 0.0905712, 0.102736])
        self.init_qpos = np.array([0.0164783, -0.080544, 0.000418251, 0.999723, 1.08014e-07, 3.21496e-07, -0.0235368, 0.314487, 0.492531, 0.01, 0.0136404])

    def step(self, a):

        x_prev,y_prev,z_prev = self.data.qpos[0:3]
        vec = self.data.qpos[0:3] - np.array([-0.015,-0.07,0])
        reward_near = np.linalg.norm(vec) #euclidiean distance
        action=np.clip(a,-1,0)
        self.do_simulation(action, self.frame_skip)
        x_next,y_next,z_next =self.data.qpos[0:3]
        reward_ctrl=np.square(action).sum()
        vel=sqrt((x_prev-x_next)**2 + (y_prev-y_next)**2 + (z_prev-z_next)**2)/self.dt
        done=False
        terminal=0

        ob = self._get_obs()

        if(reward_near!=0 and reward_near< 0.001 and vel< 0.0001):
            done=True
            terminal=10

        reward= -reward_near-0.3*reward_ctrl+terminal #penalize distance to goal, velocity and control


        return ob, reward, done, dict(reward_ctrl=reward_ctrl,reward_near=reward_near)

    def reset_model(self):

        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[7:11].flat,self.sim.data.sensordata.flat,self.data.qpos[0:2]
        ])

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy())

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):

        self.viewer = MjViewer(self.sim)
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 0.5
