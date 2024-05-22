from torch.utils.data import Dataset
from environment import * 
import os 
from simple_strategy import * 
from tqdm import tqdm 

def sample_trajectory(env):
    observations=[]
    actions=[]

    env.reset()
    strategy=SimpleStrategy(env)
    obs,info=env.reset()
    done=False 
    while done==False:
        action,info2=strategy.select_action(obs,info)
        observations.append(obs)
        actions.append(action)
        obs, reward, done, _, info=env.step(action)
    return np.asarray(observations),np.asarray(actions)

class SupervisedTrajectoryDataset(Dataset):
    def __init__(self,num_node=11):
        self.num_node=num_node 
        if os.path.exists('{}_cached_trajectories.npy'.format(num_node)):
            self.trajectories=np.load('{}_cached_trajectories.npy'.format(num_node),allow_pickle=True)
        else:
            self.trajectories=[]
            env=DecisionTreeEnv(num_node)
            for _ in tqdm(range(1000000)):
                o,a=sample_trajectory(env)
                self.trajectories.append([o,a])
            self.trajectories=np.asarray(self.trajectories)
            np.save('{}_cached_trajectories.npy'.format(num_node),self.trajectories,allow_pickle=True)
    
    def __len__(self):
        return len(self.trajectories)
    def __getitem__(self,idx):
        return (self.trajectories[idx][0],self.trajectories[idx][1])

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.from_numpy(x[0]).float() for x in sorted_batch]
        label_sequences=[torch.from_numpy(x[1]).long() for x in sorted_batch]

        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,padding_value=-1)
        label_sequences_padded=torch.nn.utils.rnn.pad_sequence(label_sequences, batch_first=True,padding_value=-1)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        return sequences_padded, lengths, label_sequences_padded

    
                
