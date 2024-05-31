from torch.utils.data import Dataset
from environment import * 
import os 
from simple_strategy import * 
from tqdm import tqdm 



class SupervisedTrajectoryDataset(Dataset):
    def __init__(self,num_train_nodes=[7,11],num_test_nodes=9,strat_class=SimpleStrategy,train=True,n_train_traj=1000000,n_test_traj=1000):
        self.num_train_nodes=num_train_nodes
        self.num_test_nodes=num_test_nodes 
        self.size=max(self.num_train_nodes)
        self.strategy_class=strat_class
        assert self.size>=self.num_test_nodes  
        self.num_actions=len(action_dict.keys())
        fname='cached_data/cached_trajectories_train_nodes_{}_test_nodes_{}_strategy_{}_train_values.npy'.format(str(self.num_train_nodes),self.num_test_nodes,self.strategy_class.__name__)
        fname2='cached_data/cached_trajectories_train_nodes_{}_test_nodes_{}_strategy_{}_test_values.npy'.format(str(self.num_train_nodes),self.num_test_nodes,self.strategy_class.__name__)
        if os.path.exists(fname):
            self.train_trajectories=np.load(fname,allow_pickle=True)
        else:
            self.train_trajectories=[]
            self.train_envs=[DecisionTreeEnv(n) for n in self.num_train_nodes]

            for _ in tqdm(range(n_train_traj)):
                env=np.random.choice(self.train_envs)
                o,a,v=self.sample_trajectory(env)
                self.train_trajectories.append([o,a,v])

            self.train_trajectories=np.asarray(self.train_trajectories)
            np.save(fname,self.train_trajectories,allow_pickle=True)
            
        if os.path.exists(fname2):
            self.test_trajectories=np.load(fname2,allow_pickle=True)
        else:
            self.test_trajectories=[]
            self.test_env=DecisionTreeEnv(self.num_test_nodes)

            for _ in tqdm(range(n_test_traj)):
                env=self.test_env 
                o,a,v=self.sample_trajectory(env)
                self.test_trajectories.append([o,a,v])

            self.test_trajectories=np.asarray(self.test_trajectories)
            np.save(fname2,self.test_trajectories,allow_pickle=True)
            
        self.obs_size=len(self.train_trajectories[0][0][0])

        self.train_mode=train 
        if self.train_mode:
            self.trajectories=self.train_trajectories 
        else:
            self.trajectories=self.test_trajectories
    
    def sample_trajectory(self,env):
        observations=[]
        actions=[]
        values=[]
        env.reset()
        strategy=self.strategy_class(env)
        obs,info=env.reset()
        done=False 
        while done==False:
            symbolic_action,env_action,info2=strategy.select_action(obs,info)
            obs=self.wrap_observation(obs,env.num_node)
            observations.append(obs)
            actions.append(symbolic_action)
            l_value=info2['left_value']
            if l_value is None:
                l_value=0.0
            elif isinstance(l_value,list):
                l_value=goodmean(l_value)
            r_value=info2['right_value']
            if r_value is None:
                r_value=0.0
            elif isinstance(r_value,list):
                r_value=goodmean(r_value)
            assert isinstance(l_value,float)
            assert isinstance(r_value,float)
            values.append([l_value,r_value])
            obs, reward, done, _, info=env.step(env_action)
        return np.asarray(observations),np.asarray(actions),np.asarray(values)

    
    def re_onehot(self,o):
        new_onehot=np.zeros(self.size)
        new_onehot[:len(o)]=o 
        return new_onehot 
    def wrap_observation(self,obs,num_node):
        inds=[num_node*i for i in range(1,6)]
        for _ in range(3):
            inds.append(inds[-1]+1)
        fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage=np.split(obs,inds)[:-1]
        fixation_onehot=self.re_onehot(fixation_onehot)
        fixation_parent_onehot=self.re_onehot(fixation_parent_onehot)
        fixation_left_child_onehot=self.re_onehot(fixation_left_child_onehot)
        fixation_right_child_onehot=self.re_onehot(fixation_right_child_onehot)
        root_node_onehot=self.re_onehot(root_node_onehot)
        return np.hstack([fixation_onehot,fixation_parent_onehot,fixation_left_child_onehot,fixation_right_child_onehot,root_node_onehot,fixation_reward,timer,stage])

    def __len__(self):
        return len(self.trajectories)
    def __getitem__(self,idx):
        return (self.trajectories[idx][0],self.trajectories[idx][1],self.trajectories[idx][2])

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.from_numpy(x[0].astype('float')).float() for x in sorted_batch]
        label_sequences=[torch.from_numpy(x[1].astype('float')).long() for x in sorted_batch]
        value_sequences=[torch.from_numpy(x[2].astype('float')).float() for x in sorted_batch]


        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,padding_value=-1234)
        label_sequences_padded=torch.nn.utils.rnn.pad_sequence(label_sequences, batch_first=True,padding_value=-1234)
        value_sequences_padded=torch.nn.utils.rnn.pad_sequence(value_sequences,batch_first=True,padding_value=-1234)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        return sequences_padded, lengths, label_sequences_padded,value_sequences_padded
        #return sequences_padded, lengths, label_sequences_padded

    
                

