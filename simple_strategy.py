import numpy as np 
class SimpleStrategy():
    def __init__(self,env):
        self.env=env 
        self.value_buffer=[]
        self.left_value=None 
        self.right_value=None 
        self.info_gathering_stage=0 #0=gather left info, 1=gather right info, 2=execute plan 
        self.n_nodes=self.env.num_node 
    def select_action(self,obs,info):
        inds=[self.n_nodes*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        if self.info_gathering_stage==0:
            if np.sum(fixation_left_child_onehot)<1:
                self.value_buffer.append(fixation_reward)
                self.left_value=np.sum(self.value_buffer)
                action=np.argmax(root_node_onehot)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                self.info_gathering_stage=1 
                self.value_buffer=[]
                return action,info_dict
            else:
                self.value_buffer.append(fixation_reward)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                action=np.argmax(fixation_left_child_onehot)
                return action,info_dict 
        if self.info_gathering_stage==1:
            if np.sum(fixation_right_child_onehot)<1:
                self.value_buffer.append(fixation_reward)
                self.right_value=np.sum(self.value_buffer)
                action=self.n_nodes*2 #Turn off fixation stage. 
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                self.info_gathering_stage=2
                self.value_buffer=[]
                return action,info_dict 
            else:
                self.value_buffer.append(fixation_reward)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                action=np.argmax(fixation_right_child_onehot)
                return action,info_dict 
        if self.info_gathering_stage==2:
            assert self.left_value is not None and self.right_value is not None 
            if self.right_value>self.left_value:
                action=self.n_nodes+np.argmax(fixation_right_child_onehot)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict 
            else:
                action=self.n_nodes+np.argmax(fixation_left_child_onehot)
                info_dict={'value_buffer':np.array(self.value_buffer),'left_value':self.left_value,'right_value':self.right_value,'info_gathering_stage':self.info_gathering_stage}
                return action,info_dict
            
