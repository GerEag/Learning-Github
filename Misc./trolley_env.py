#! /usr/bin/env python



#######################################################
#
#
# trolley_env.py
#
#
# openai gym environment for a planar crane with a simple pendulum
#
#
# Gerald Eaglin, ULL, 1/29/2020
#
#######################################################


import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.integrate import solve_ivp
import numpy as np
from numpy import sin, cos, clip, array



class TrolleyEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        print('\n\nRunning environment for a trolley.\n\n')
        self.mass = 1.0 # kg
        self.gravity = 9.8 # m/s^2
        self.wn = np.pi # rad/s

        # define natural frequency and use it to calculate cable length
        self.cable_length = self.gravity/self.wn**2 # m
        self.tau = 0.02  # seconds between state updates
        self.t = 0.0 # current run time of the episode
        self.time_limit = 10 # end the episode after 10 seconds

        # startpoint limits
        self.min_set = -5
        self.max_set = 5

        # threshold to fail the episode
        self.x_threshold = 1.5*self.max_set
        self.th_threshold = 60 * np.pi/180 # +/- 60 degrees

        # reward function weights
        self.a = 1 # position weight
        self.b = 1 # angle weight
        self.c = 1
        self.d = 1
        self.e = 1
        self.f = 1

        action_bound = 1.0 # max velocity input
        # bounds on action space
        self.ac_high = action_bound
        self.ac_low = -action_bound

        # bounds on observation space
        self.ob_high = np.array([
                    1.5*self.x_threshold,       # x displacement
                    np.finfo(np.float32).max,   # x velocity
                    # 1.5 * self.th_threshold,    # angular displacement
                    # np.finfo(np.float32).max
                    ])  # angular velocity

        self.ob_low = np.array([
                    -1.5*self.x_threshold,      # x displacement
                    np.finfo(np.float32).min,   # x velocity
                    # -1.5 * self.th_threshold,    # angular displacement
                    # np.finfo(np.float32).min
                    ])  # angular velocity


        self.action_space = spaces.Box(low=self.ac_low, high=self.ac_high, shape=(1,))
        self.observation_space = spaces.Box(low=self.ob_low, high=self.ob_high)

        self.seed()
        self.viewer = None
        self.state = None
        self.done = False
        self.reset()

        self.steps_beyond_done = None

    def eq_of_motion(self,t,w,p):

        # pos, vel, th, th_dot = w
        # acc, g, l = p
        pos, vel = w
        acc = p


        # sys_eq = [x_dot,
        #             x_acc,
        #             th_dot,
        #             (-g*sin(th) - cos(th) * x_acc)/l
        #             ]

        # The trolley is acceleration controlled; integrate here
        sys_eq = [
                vel,
                acc,
                # th_dot,
                # (-g*sin(th) - cos(th) * acc)/l
                ]


        return sys_eq


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # pass the action to the environment and determine how the system responds
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # action is the acceleration command for the trolley
        acc = action


        # initial values
        X0 = [self.state[0], self.state[1]]

        p = acc
        # solve the state space equations of motion
        resp = solve_ivp(lambda t,w: self.eq_of_motion(t,w,p), [self.t, self.t+self.tau], X0, t_eval=[self.t,self.t+self.tau])

        # x = resp.y[0,-1]
        # x_dot = resp.y[1,-1]
        # th = resp.y[2,-1]
        # th_dot = resp.y[3,-1]
        # self.state = array([x, x_dot, th, th_dot])

        pos = resp.y[0,-1]
        vel = resp.y[1,-1]
        # th = resp.y[2,-1]
        # th_dot = resp.y[3,-1]

        # assume trolley velocity is equal to vel_com
        self.state = array([pos, vel])

        # reward function should minimize position error and angular displacement
        # reward = -x**2 - b*th**2 # time is not necessary because it is independent of the action

        reward = -self.a*(pos/self.max_set)**2 #- self.b*(th/self.th_threshold)**2
        # reward = -self.a*(pos/self.max_set)**2 - self.b*(th/self.th_threshold)**2 # time is not necessary because it is independent of the action
        # if x > self.x_threshold or x < -self.x_threshold:
        #     reward += -(np.abs(x)-self.x_threshold)**2
        # if th > self.th_threshold or th < -self.th_threshold:
        #     reward += -(np.abs(th)-self.th_threshold)**2
        # if vel_com > self.ac_high or vel_com < self.ac_low:
        #     reward += -(np.abs(vel_com)-self.ac_high)**2
        # if x_acc > 10 or x_acc < -10:
        #     reward += -(np.abs(x_acc)-10)**2


        self.t += self.tau
        if self.t >= self.time_limit: # check the episode time limit
            self.done = True

        return self.state, np.float(reward), self.done, {}


    def reset(self):
        self.done = False
        self.t = 0.0 # restart the time
        pos = self.np_random.uniform(low=self.min_set, high=self.max_set) # randomize the initial crane position
        self.state = array([pos, 0.0])
        self.steps_beyond_done = None # what does this do?
        return self.state

    def render(self, mode='human', close=False):
        screen_width = 600 # size of the window
        screen_height = 400

        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        # carty = 100 # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * 1.0
        # cartwidth = 50.0
        # cartheight = 30.0
        world_width = 1.5*self.x_threshold
        # scale = 50
        scale = screen_width/world_width

        # trolley_width = 20
        # trolley_height = 20
        trolley_width = 0.5*scale
        trolley_height = 0.5*scale
        trolley_y = 3/4 * screen_height # y-position of trolley in screen


        cable_width = 2
        cable_len = self.cable_length*scale
        payload_size = 0.1*scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height) # set screen size

            # add a polygon representing the trolley
            l,r,t,b = -trolley_width/2, trolley_width/2, trolley_height/2, -trolley_height/2 # defines boundaries of shape measured from center
            trolley = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) # corners of the polygon
            self.trolleytrans = rendering.Transform() # what does this do?
            trolley.add_attr(self.trolleytrans)
            self.viewer.add_geom(trolley) # add trolley shape to viewer

            # # add a polygon representing the cable
            # l,r,t,b = -cable_width/2, cable_width/2, 0.0, -self.cable_length*scale # defines boundaries of shape measured from center
            # cable = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) # corners of the polygon
            # self.cabletrans = rendering.Transform() # what does this do?
            # cable.add_attr(self.cabletrans)
            # self.viewer.add_geom(cable)

            # # add a circle for the payload
            # payload = rendering.make_circle(payload_size)
            # payload.set_color(1,0,0)
            # self.payloadtrans = rendering.Transform()
            # payload.add_attr(self.payloadtrans)
            # self.viewer.add_geom(payload)

            # add horizontal line along which trolley travels
            self.track = rendering.Line(((self.min_set*scale)+ screen_width/2,trolley_y), ((self.max_set*scale)+screen_width/2,trolley_y))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            # add vertical line marking zero displacement
            self.origin = rendering.Line((screen_width/2.0,(1/2)*screen_height), (screen_width/2,(5/6)*screen_height))
            self.origin.set_color(0,0,0)
            self.viewer.add_geom(self.origin)


        if self.state is None: return None

        pos, vel = self.state
        # trolley motion
        trolley_x = pos*scale + screen_width/2.0 # x-position of the trolley
        self.trolleytrans.set_translation(trolley_x, trolley_y) # sets and updates location of cart

        # # cable motion
        # self.cabletrans.set_translation(trolley_x, trolley_y)
        # self.cabletrans.set_rotation(th)

        # # payload motion
        # payload_x = (pos + self.cable_length*sin(th))*scale + screen_width/2
        # payload_y = trolley_y - self.cable_length*cos(th)*scale
        # self.payloadtrans.set_translation(payload_x, payload_y)

        # self.settrans.set_translation(self.setpoint*scale + screen_width/2, mass_y) # sets and updates location of setpoint

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()