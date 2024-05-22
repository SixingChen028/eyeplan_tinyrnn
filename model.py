import torch
import torch.nn as nn 

class TinyRNN(nn.Module):
    def __init__(self,input_size=58,hidden_size=128,num_actions=23):
        super(TinyRNN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size  
        self.num_action=num_actions
        self.rnn=nn.GRU(self.input_size,self.hidden_size,batch_first=True)
        self.y_out=nn.Linear(self.hidden_size,self.num_action)
    
    def forward(self,task_seq,device,hidden=None):
        if hidden is None:
            hidden=torch.zeros(1,task_seq.shape[0],self.hidden_size)
        hidden=hidden.to(device)
        lstm_out,hidden=self.rnn(task_seq,hidden)
        pred=self.y_out(lstm_out)
        return pred,hidden 


    
