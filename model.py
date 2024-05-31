import torch
import torch.nn as nn 
import torch.nn.init as init 
import math 
from torch import Tensor
import torch.nn.functional as F

class ElementWiseLinear(nn.Module):

    def __init__(self, num_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features=num_features
        self.weight = nn.Parameter(torch.empty((num_features,1), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_features,1), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return self.weight[:,0]*input+self.bias[:,0]

    def extra_repr(self) -> str:
        return f'features={self.num_features}, bias={self.bias is not None}'
    


class TinyRNN(nn.Module):
    def __init__(self,input_size=58,num_actions=23):
        super(TinyRNN,self).__init__()
        self.input_size=input_size
        self.num_action=num_actions
        #self.hidden_size=2
        self.hidden_size=self.num_action
        #self.hidden_size=256
        print("TinyRNN Specs: GRU with input size {} and hidden dim {}".format(self.input_size,self.hidden_size))
        self.rnn=nn.GRU(self.input_size,self.hidden_size,batch_first=True)
        self.pi_out=ElementWiseLinear(self.hidden_size)
        #self.pi_out=nn.Linear(self.hidden_size,self.num_action)
        #self.value_out=nn.Linear(self.hidden_size,2)
    
    def forward(self,task_seq,device,hidden=None):
        if hidden is None:
            hidden=torch.zeros(1,task_seq.shape[0],self.hidden_size)
        hidden=hidden.to(device)
        lstm_out,hidden=self.rnn(task_seq,hidden)
        action_pred=self.pi_out(lstm_out)
        return action_pred,hidden 
        #value_pred=self.value_out(lstm_out)
        #return action_pred,value_pred,hidden 
    


    
