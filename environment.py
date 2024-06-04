import numpy as np
import time
import os

import torch
import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from graph import *


class DecisionTreeEnv(gym.Env):
    """
    A decision tree environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_node,
            t_max = 100,
            shuffle_nodes = True,
        ):
        """
        Construct an environment.
        """

        self.t_max = t_max
        self.num_node = num_node
        self.shuffle_nodes = shuffle_nodes
        self.reward_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8])

        # initialize graph
        self.graph = Graph(self.num_node, self.reward_set)

        # initialize action space
        self.action_space = Discrete(self.num_node + self.num_node + 1)

        # initialize observation space
        observation_shape = (
            self.num_node + # fixation node (num_node,)
            self.num_node * 3 + # parent and childs of fixation node (3 * num_node,)
            self.num_node + # root_node (num_node,)
            3, # fixation reward, timer, stage
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


    def reset(self, seed = None, option = {}):
        """
        Reset the environment.
        """

        # reset the trial
        self.init_trial()

        # get parent and child nodes
        fixation_parent_node = self.graph.predecessors(self.fixation_node)
        fixation_child_nodes = self.graph.successors(self.fixation_node)

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_node, labels = self.fixation_node),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_parent_node),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_child_nodes[0]),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_child_nodes[1]),
            self.one_hot_coding(num_classes = self.num_node, labels = self.graph.root_node),
            self.graph.rewards[self.fixation_node],
            self.timer,
            self.stage,
        ])
    
        # get info
        info = {
            'fixation_node': self.fixation_node,
            'fixation_reward': self.graph.rewards[self.fixation_node],
            'fixation_gain': self.graph.gains[self.fixation_node],
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        self.timer += 1
        done = False
        reward = 0.

        action_type = action // self.num_node
        action_node = action % self.num_node

        # fixation stage
        if self.stage == 0:
            # fixation action
            if action_type == 0:
                # move fixation to the node
                self.fixation_node = action_node

                # update record
                updated = self.update_record(action_node)

            # decision action (debugging use)
            elif action_type == 1:
                # raise ValueError('Execute decision action in fixation stage.')
                pass

            # stage switch action
            elif action_type == 2:
                # switch to decision stage
                self.stage = 1

                # reset fixation to root node
                #self.fixation_node = self.graph.root_node

        # decision stage
        elif self.stage == 1:
            # fixation action (debugging use)
            if action_type == 0:
                # raise ValueError('Execute fixation action in decision stage.')
                pass

            # decision actionf
            elif action_type == 1:
                # move fixation to the node
                self.fixation_node = action_node
                done=True 
            
            # stage switch action (debugging use)
            elif action_type == 2:
                # raise ValueError('Execute stage swtich action in decision stage.')
                pass
        
            # reward calculation
            if self.fixation_node in self.graph.leaf_nodes and self.graph.gains[self.fixation_node] == np.max(self.graph.gains[self.graph.leaf_nodes]):
                reward = 1.

        # end a trial
        #if (self.stage == 1 and self.fixation_node in self.graph.leaf_nodes) or self.timer == self.t_max:
        #    done = True

        # get parent and child nodes
        fixation_parent_node = self.graph.predecessors(self.fixation_node)
        fixation_child_nodes = self.graph.successors(self.fixation_node)

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_node, labels = self.fixation_node),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_parent_node),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_child_nodes[0]),
            self.one_hot_coding(num_classes = self.num_node, labels = fixation_child_nodes[1]),
            self.one_hot_coding(num_classes = self.num_node, labels = self.graph.root_node),
            self.graph.rewards[self.fixation_node],
            self.timer,
            self.stage,
        ])
    
        # get info
        info = {
            'fixation_node': self.fixation_node,
            'fixation_reward': self.graph.rewards[self.fixation_node],
            'fixation_gain': self.graph.gains[self.fixation_node],
            'stage': self.stage,
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize timer and stage
        self.timer = 0
        self.stage = 0

        # initialize the tree
        self.graph.reset(shuffle_nodes = self.shuffle_nodes)

        # initialize fixation record
        self.init_record()

        # initialize fixation node ot root node
        self.fixation_node = self.graph.root_node
    

    def init_record(self):
        """
        Initialize fixation record.
        """

        # initialize node recordings
        self.visited_nodes = np.array([self.graph.root_node])
        self.candidate_nodes = np.array(self.graph.child_dict[self.graph.root_node])
        self.valid_nodes = np.union1d(self.visited_nodes, self.candidate_nodes)

    
    def update_record(self, node):
        """
        Update fixation record.
        """

        updated = 0

        # if fixating a new candiate node
        if node in self.candidate_nodes:
            updated = 1

            # include the new node into visited nodes
            self.visited_nodes = np.append(self.visited_nodes, node)

            # remove the new node from candidate nodes
            self.candidate_nodes = np.delete(self.candidate_nodes, np.where(self.candidate_nodes == node)[0])

            # if the new node has child nodes, add them into candidate nodes
            if node in self.graph.child_dict.keys():
                self.candidate_nodes = np.append(self.candidate_nodes, self.graph.child_dict[node])
                    
        # update valid nodes
        self.valid_nodes = np.union1d(self.visited_nodes, self.candidate_nodes)

        return updated


    def get_action_mask(self):
        """
        Get action mask.
        """

        mask = torch.zeros((1, self.action_space.n), dtype = torch.bool) # no batch training (batch_size = 1)

        # fixation stage
        if self.stage == 0:
            mask[0, self.valid_nodes] = True
            mask[0, -1] = True

        # decision stage
        elif self.stage == 1:
            mask[0, self.num_node + self.graph.leaf_nodes] = True
        
        return mask


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.num_node = env.get_wrapper_attr('num_node')
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        # initialize previous variables
        self.init_prev_variables()

        # define new observation space
        new_observation_shape = (
            self.env.observation_space.shape[0] + # obs
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        # concatenate previous variables into observation
        obs_wrapped = self.wrap_obs(obs)

        # update previous variables
        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        # initialize previous physical action and reward
        self.init_prev_variables()

        # concatenate previous physical action and reward into observation
        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs, # current obs
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward,
        ])
        return obs_wrapped



if __name__ == '__main__':
    # testing
    
    env = DecisionTreeEnv(num_node = 5, t_max = 50)
    env = MetaLearningWrapper(env)

    for i in range(50):

        obs, info = env.reset()
        done = False

        print('connctions:', env.graph.child_dict)
        print('gains:', env.graph.gains)
        print('initial obs:', obs.shape)
        
        while not done:

            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            print(
                'obs:', obs.shape, '|',
                'action:', action, '|',
                'reward:', np.round(reward, 3), '|',
                'stage:', env.stage, '|',
                'fixation node:', env.fixation_node, '|',
                'done:', done, '|',
                'timer:', env.timer, '|'
            )
        print()

