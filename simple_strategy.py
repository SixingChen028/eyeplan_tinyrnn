import numpy as np 
from enum import Enum 


def goodmean(x):
    if len(x)==0:
        return 0.0
    else:
        return np.mean(x)
    
class SymbolicActionSpace(Enum):
    FIXATE_LEFT=0
    FIXATE_RIGHT=1
    FIXATE_ROOT=2
    SWITCH_PLAN=3
    PHYSICAL_LEFT=4
    PHYSICAL_RIGHT=5



action_dict={SymbolicActionSpace.FIXATE_LEFT:0,SymbolicActionSpace.FIXATE_RIGHT:1,SymbolicActionSpace.FIXATE_ROOT:2,SymbolicActionSpace.SWITCH_PLAN:3,SymbolicActionSpace.PHYSICAL_LEFT:4,SymbolicActionSpace.PHYSICAL_RIGHT:5}

class SimpleStrategy():
    def __init__(self,env):
        self.env=env 
        self.value_buffer=[]
        self.left_value=None 
        self.right_value=None 
        self.info_gathering_stage=0 #0=gather left info, 1=gather right info, 2=execute plan 
        self.n_nodes=self.env.num_node 
        self.reduced_action_space=[]
    def select_action_symbolic_space(self,obs,info):
        inds=[self.n_nodes*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        if self.info_gathering_stage==0:
            if np.sum(fixation_left_child_onehot)<1:
                self.value_buffer.append(fixation_reward)
                self.left_value=np.sum(self.value_buffer)
                #action=np.argmax(root_node_onehot)
                action=SymbolicActionSpace.FIXATE_ROOT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                self.info_gathering_stage=1 
                self.value_buffer=[]
                return action,info_dict
            else:
                self.value_buffer.append(fixation_reward)
                self.left_value=np.sum(self.value_buffer)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                #action=np.argmax(fixation_left_child_onehot)
                action=SymbolicActionSpace.FIXATE_LEFT
                return action,info_dict 
        if self.info_gathering_stage==1:
            if np.sum(fixation_right_child_onehot)<1:
                self.value_buffer.append(fixation_reward)
                self.right_value=np.sum(self.value_buffer)
                #action=self.n_nodes*2 #Turn off fixation stage. 
                action=SymbolicActionSpace.SWITCH_PLAN
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                self.info_gathering_stage=2
                self.value_buffer=[]
                return action,info_dict 
            else:
                self.value_buffer.append(fixation_reward)
                self.right_value=np.sum(self.value_buffer)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                #action=np.argmax(fixation_right_child_onehot)
                action=SymbolicActionSpace.FIXATE_RIGHT
                return action,info_dict 
        if self.info_gathering_stage==2:
            assert self.left_value is not None and self.right_value is not None 
            if self.right_value>self.left_value:
                #action=self.n_nodes+np.argmax(fixation_right_child_onehot)
                action=SymbolicActionSpace.PHYSICAL_RIGHT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict 
            else:
                #action=self.n_nodes+np.argmax(fixation_left_child_onehot)
                action=SymbolicActionSpace.PHYSICAL_LEFT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict
            
    def select_action(self,obs,info):
        inds=[self.n_nodes*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        action,info_dict=self.select_action_symbolic_space(obs,info)
        symbolic_action=action_dict[action]
        env_action_space=[np.argmax(fixation_left_child_onehot),np.argmax(fixation_right_child_onehot),np.argmax(root_node_onehot),self.n_nodes*2,self.n_nodes+np.argmax(fixation_left_child_onehot),self.n_nodes+np.argmax(fixation_right_child_onehot)]
        return symbolic_action,env_action_space[symbolic_action],info_dict

            
class RolloutStrategy():
    def __init__(self,env,num_rollouts=2):
        self.env=env 
        self.rolling_out=None 
        self.value_buffer=[]
        self.left_value=[]
        self.right_value=[] 
        self.info_gathering_stage=0 #0=rolling out 1=execute plan 
        self.n_nodes=self.env.num_node 
        self.reduced_action_space=[]
        self.num_rollouts=num_rollouts

    def select_action_symbolic_space(self,obs,info):
        inds=[self.n_nodes*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        if self.info_gathering_stage==0:
            if self.rolling_out is None:
                self.rolling_out='left'
            
            if np.sum(fixation_parent_onehot)<1: #at the root. 
                self.value_buffer.append(fixation_reward)
                if self.rolling_out=='left':
                    action=SymbolicActionSpace.FIXATE_LEFT
                else:
                    action=SymbolicActionSpace.FIXATE_RIGHT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict 

            possible_actions=[] 
            if np.sum(fixation_left_child_onehot)>0:
                possible_actions.append(0)
            if np.sum(fixation_right_child_onehot)>0:
                possible_actions.append(1)
            if len(possible_actions)==0: #At a leaf
                if self.rolling_out=='left':
                    self.value_buffer.append(fixation_reward)
                    self.left_value.append(np.sum(self.value_buffer))
                    info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                    self.value_buffer=[]
                else:
                    self.value_buffer.append(fixation_reward)
                    self.right_value.append(np.sum(self.value_buffer))
                    info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                    self.value_buffer=[]
                
                if len(self.left_value)+len(self.right_value)>=self.num_rollouts:
                    self.info_gathering_stage=1
                    action=SymbolicActionSpace.SWITCH_PLAN 
                    return action,info_dict 
                else:
                    if self.rolling_out=='left':
                        self.rolling_out='right'
                    else:
                        self.rolling_out='left'
                    action=SymbolicActionSpace.FIXATE_ROOT
                    return action,info_dict 
            else: #non-leaf non-root 
                action=np.random.choice(possible_actions)
                if action==0:
                    self.value_buffer.append(fixation_reward)
                    info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                    action=SymbolicActionSpace.FIXATE_LEFT
                    return action,info_dict 
                else:
                    self.value_buffer.append(fixation_reward)
                    info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                    action=SymbolicActionSpace.FIXATE_RIGHT
                    return action,info_dict 
                
        if self.info_gathering_stage==1:
            assert self.left_value is not None and self.right_value is not None 
            if goodmean(self.right_value)>goodmean(self.left_value):
                #action=self.n_nodes+np.argmax(fixation_right_child_onehot)
                action=SymbolicActionSpace.PHYSICAL_RIGHT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict 
            else:
                action=SymbolicActionSpace.PHYSICAL_LEFT
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict
            
            
    def select_action(self,obs,info):
        inds=[self.n_nodes*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        action,info_dict=self.select_action_symbolic_space(obs,info)
        symbolic_action=action_dict[action]
        env_action_space=[np.argmax(fixation_left_child_onehot),np.argmax(fixation_right_child_onehot),np.argmax(root_node_onehot),self.n_nodes*2,self.n_nodes+np.argmax(fixation_left_child_onehot),self.n_nodes+np.argmax(fixation_right_child_onehot)]
        return symbolic_action,env_action_space[symbolic_action],info_dict
    
