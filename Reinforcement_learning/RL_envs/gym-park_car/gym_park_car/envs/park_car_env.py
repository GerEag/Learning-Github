#! /usr/bin/env python 3

#######################################################
#
#
# park_car_env.py
#
#
# openai gym environment for parking a car
#
#
# Gerald Eaglin, ULL, 1/20/2023
#
#######################################################

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import numpy as np
from numpy import sin, cos, tan, clip, array

# TODO: make changes from "render" on down

class ParkCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,velocity_limit=10,steering_limit=np.pi/4):
        print('\n\nRunning environment for parking a car.\n\n')

        self.velocity_limit = velocity_limit
        self.steering_limit = steering_limit

        self.length = 5 # length of the car

        self.angular_velocity_limit = self.velocity_limit*np.tan(self.steering_limit)/self.length

        self.tau = 0.05  # seconds between state updates
        self.t = 0.0 # current run time of the episode
        self.time_limit = 30 # end the episode after 5 seconds
        
        self.num_states = 3 # used for evaluation purposes
        self.num_inputs = 2 # used for evaluation purposes

        # workspace bounds
        self.workspace_width = 50 # width of workspace
        self.workspace_height = 50 # height of workspace

        # bounds on action space
        # self.ac_high = 1 # normalize action space
        # self.ac_low = -1 # normalize action space
        # TODO: I should maybe normalize these actions
        self.ac_high = np.array([
                                 self.velocity_limit,
                                 self.steering_limit,
                                 ])
        
        self.ac_low = np.array([
                                 -self.velocity_limit,
                                 -self.steering_limit,
                                 ])


        # # reward function weights
        # self.x_weight = (1/self.max_pos)**2 # position weight (this is arbitrary right now)
        # self.th_weight = (1/self.max_th)**2 # angle weight

        # bounds on observation space
        self.ob_high = np.array([
                                 self.workspace_width/2,       # x position
                                 self.workspace_height/2,      # y position
                                 np.finfo(np.float32).max,   # car rotation
                                #  self.velocity_limit,        # x velocity
                                #  self.velocity_limit,        # y velocity
                                #  self.angular_velocity_limit # angular_velocity
                                 ])

        self.ob_low = np.array([
                                -self.workspace_width/2,     # x position
                                -self.workspace_height/2,    # y position
                                np.finfo(np.float32).min,    # angular displacement
                                # -self.velocity_limit,        # x velocity
                                # -self.velocity_limit,        # y velocity
                                # -self.angular_velocity_limit # angular_velocity
                                ])

        self.action_space = spaces.Box(low=self.ac_low, high=self.ac_high)
        self.observation_space = spaces.Box(low=self.ob_low, high=self.ob_high)

        self.seed()
        self.viewer = None
        self.state = None
        self.done = False
        # self.terminated = False
        # self.truncated = False
        self.reset()

        self.steps_beyond_done = None

        # alias class method corresponding to desired input type
        # self.alias_controller()


    def eq_of_motion(self,t,w,velocity_command,steering_command):

        x, y, th_h = w

        sys_eq = [cos(th_h)*velocity_command,
                  sin(th_h)*velocity_command,
                  (tan(steering_command)*velocity_command)/self.length
                  ]

        return sys_eq

    def normalize_observation(self,x,x_dot,th,th_dot):
        """Normalize the observations between [-1,1] according to the max and min observation space"""

        x = 2 * ((x-self.min_pos)/(self.max_pos-self.min_pos)) - 1
        x_dot = 2 * ((x_dot-self.min_velocity)/(self.max_velocity-self.min_velocity)) - 1
        th = 2 * ((th-self.min_th)/(self.max_th-self.min_th)) - 1
        th_dot = 2 * ((th_dot-self.min_th_dot)/(self.max_th_dot-self.min_th_dot)) - 1

        return x, x_dot, th, th_dot

    def denormalize_observation(self,x,x_dot,th,th_dot):
        """Denormalize the observations back to physical scale"""

        x = 0.5 * (x + 1) * (self.max_pos - self.min_pos) + self.min_pos
        x_dot = 0.5 * (x_dot + 1) * (self.max_velocity - self.min_velocity) + self.min_velocity
        th = 0.5 * (th + 1) * (self.max_th - self.min_th) + self.min_th
        th_dot = 0.5 * (th_dot + 1) * (self.max_th_dot - self.min_th_dot) + self.min_th_dot

        return x, x_dot, th, th_dot

    def pure_RL_input(self,norm_action):
        '''Acceleration input to the system is from the agent alone'''

        action = self.action_bound*float(norm_action) # action is normalized from -1 to 1
        total_input = action
        self.inputs = [action] # should this be updated after clipping the action?

        return total_input

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, norm_action):
        # pass the action to the environment and determine how the system responds
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        x, y, th_h = self.state

        velocity_command, steering_command = self.clip_control_output(norm_action,self.state)

        # initial values
        X0 = [x, y, th_h]

        # extra parameters
        p = [velocity_command, steering_command]
        # solve the state space equations of motion
        resp = solve_ivp(self.eq_of_motion, [self.t, self.t+self.tau], X0, t_eval=[self.t,self.t+self.tau], args=p)

        x = resp.y[0,-1]
        y = resp.y[1,-1]
        th_h = resp.y[2,-1]

        self.state = array([x, y, th_h])
        # apply state constraints
        self.state = self.clip_state(self.state)

        reward = -(x**2 + y**2) # distance from the origin

        # # normalize state/observations
        # norm_x, norm_x_dot, norm_th, norm_th_dot = self.normalize_observation(x,x_dot,th,th_dot)

        # # define normalized states (include time elapsed in episode)
        # norm_state = array([norm_x, norm_x_dot, norm_th, norm_th_dot])

        self.t += self.tau
        if self.t >= self.time_limit: # check the episode time limit
            self.done = True
            # self.truncated = True
            self.info["TimeLimit.truncated"] = True

        return self.state, np.float(reward), self.done, self.info
        # return norm_state, np.float(reward), self.terminated, self.truncated, {}


    def reset(self):
        self.done = False
        self.IN_STABLE_REGION = False
        self.info = {}
        # self.truncated = False
        self.t = 0.0 # restart the time

        x = self.np_random.uniform(low=-self.workspace_width/2,high=self.workspace_width/2)
        y = self.np_random.uniform(low=-self.workspace_height/2,high=self.workspace_height/2)
        th_h = self.np_random.uniform(low=-np.pi,high=np.pi)

        self.state = array([x,y,th_h]) # start position is pi for now (stable equil.)
        # norm_state = self.normalize_observation(*self.state)
        self.steps_beyond_done = None # what does this do?
        return self.state

    def render(self, mode='human', close=False):
        
        # define pixel size of the window
        screen_width = 600
        screen_height = 600

        # define size of "physical" world shown by window
        world_width = 1.1*self.workspace_width # meters
        scale = screen_width/world_width # pixels/meter

        car_length = self.length*scale
        car_width = 0.25*car_length

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height) # set screen size

            # TODO: add a dot on the center of the car

            # add a polygon representing the car
            l,r,t,b = -car_width/2, car_width/2, car_length/2, -car_length/2 # defines boundaries of shape measured from center of car
            cable = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) # corners of the polygon
            self.cabletrans = rendering.Transform()
            cable.add_attr(self.cabletrans)
            self.viewer.add_geom(cable)

            # # add a circle for the payload
            # payload = rendering.make_circle(payload_size)
            # payload.set_color(1,0,0)
            # self.payloadtrans = rendering.Transform()
            # payload.add_attr(self.payloadtrans)
            # self.viewer.add_geom(payload)

            # # add horizontal line along which trolley travels
            # self.track = rendering.Line(((self.min_pos*scale)+ screen_width/2,trolley_y), ((self.max_pos*scale)+screen_width/2,trolley_y))
            # # self.track = rendering.Line(((-5*scale)+ screen_width/2,trolley_y), ((5*scale)+screen_width/2,trolley_y))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

            # # add vertical line marking zero displacement
            # self.origin = rendering.Line((screen_width/2.0,0.5*trolley_y), (screen_width/2,1.5*trolley_y))
            # self.origin.set_color(0,0,0)
            # self.viewer.add_geom(self.origin)


        if self.state is None: return None

        x, x_dot, th, th_dot = self.state
        # trolley motion
        trolley_x = x*scale + screen_width/2.0 # x-position of the trolley (in pixels)
        self.trolleytrans.set_translation(trolley_x, trolley_y) # sets and updates location of cart

        # cable motion
        self.cabletrans.set_translation(trolley_x, trolley_y)
        self.cabletrans.set_rotation(th+np.pi) # sets angle measured from downward vertical

        # payload motion
        # payload_x = (x - self.cable_length*sin(th))*scale + screen_width/2
        # payload_y = trolley_y + self.cable_length*cos(th)*scale
        payload_x = (x*scale - cable_len*sin(th)) + screen_width/2
        payload_y = trolley_y + cable_len*cos(th)
        self.payloadtrans.set_translation(payload_x, payload_y)

        # self.settrans.set_translation(self.setpoint*scale + screen_width/2, mass_y) # sets and updates location of setpoint

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def change_state(self,state):

        self.state = state
        # norm_obs = self.normalize_observation(*self.state)
        norm_obs = self.state

        return norm_obs, state

    def clip_control_output(self,unclipped_output,state):

        x, y, th_h = state

        velocity_command = clip(unclipped_output[0],-self.velocity_limit,self.velocity_limit)
        steering_command = clip(unclipped_output[1],-self.steering_limit,self.steering_limit)

        # if at maximum position, don't move
        if x >= self.workspace_width/2 and velocity_command*cos(th_h) > 0: # right side of screen and moving right
            velocity_command = 0
        elif x <= -self.workspace_width/2 and velocity_command*cos(th_h) < 0: # left side of screen and moving left
            velocity_command = 0
        elif y >= self.workspace_height/2 and velocity_command*sin(th_h) > 0: # top of screen and moving up
            velocity_command = 0
        elif y <= -self.workspace_height/2 and velocity_command*sin(th_h) < 0: # bottom of screen and moving down
            velocity_command = 0

        return velocity_command, steering_command

    def clip_state(self,state):

        x, y, th_h = state

        # clip the car position
        x = clip(x,-self.workspace_width/2,self.workspace_width/2)
        y = clip(y,-self.workspace_height/2,self.workspace_height/2)

        state = array([x,y,th_h])

        return state

    def close(self):
        if self.viewer: self.viewer.close()
