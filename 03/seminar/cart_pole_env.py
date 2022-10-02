#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:43:52 2022

@author: strike
"""
import random
import numpy as np

class Env:
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1
        self.pole_mass = 0.1
        self.half_pole_length = 0.5
        self.max_force = 10
        self.ts = 0.02
        self.angle_theshold = 12*3.14/180
        self.displasmant_thrshold = 2.4
        self.reward_not_fall = 1
        self.penalty = -10
        self.state = np.array([2*0.05*np.random.rand() - 0.05,0,0,0])
        self.force = 10
        
    def get_observation(self):
        return self.state
    
    def step(self,action):
        if action == -1:
            self.force = - self.force
            
        CosTheta = np.cos(self.state[0])
        SinTheta = np.sin(self.state[0])
        SystemMass = self.cart_mass + self.pole_mass
        temp = (self.force + self.pole_mass*self.half_pole_length*self.state[1]**2*SinTheta)/SystemMass
            
        theta_dot_dot = (self.gravity*SinTheta - CosTheta*temp)/(self.half_pole_length*(4.0/3.0-self.pole_mass*CosTheta**2/SystemMass))
                
        x_dot_dot = temp - self.pole_mass*self.half_pole_length*theta_dot_dot*CosTheta/SystemMass
        
        self.state[1] = self.state[1] + self.ts*theta_dot_dot
        self.state[0] = self.state[0] + self.ts*self.state[1]
        self.state[3] = self.state[3] + self.ts*x_dot_dot
        self.state[2] = self.state[2] + self.ts*self.state[3]
        
        self.is_done = bool(np.abs(self.state[0])>self.angle_theshold or np.abs(self.state[2])>self.displasmant_thrshold)
        if self.is_done:
            reward = self.penalty
        else:
            reward = self.reward_not_fall
            
        return reward
    
    def reset(self):
        self.state = np.array([2*0.5*np.random.rand() - 0.05,0,0,0])
        
    
    
t = Env()
obs = t.get_observation()
t.step(1)