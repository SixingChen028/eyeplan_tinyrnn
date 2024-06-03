from utils import log
from model import TinyRNN
from dataset import * 
import torch 
import torch.nn as nn 
import time 
from torch.utils.data import DataLoader,random_split 
import torch.optim as optim
import sys 

def mse_loss(input, target, ignored_index=-1234, reduction='mean'):
    mask = target != ignored_index    
    out = ((input*mask)-(target*mask))**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


def train(model,device,optimizer,epoch,train_loader,log_interval=10):
    log.info("Starting epoch {}".format(epoch))
    model.train()
    model.to(device)
    weights_actions=torch.tensor([0.1,0.1,0.1,0.1,1.0,1.0]).float().cuda()
    policy_loss_fn=nn.CrossEntropyLoss(weight=weights_actions,ignore_index=-1234)
    value_loss_fn=mse_loss
    for batch_idx,batch_data in enumerate(train_loader):
        start_time = time.time()
        seq,seq_lengths,labels,values=batch_data  
        #seq,seq_lengths,labels=batch_data
        labels=labels.to(device)
        seq=seq.to(device)
        values=values.to(device)

        optimizer.zero_grad() 
        #pred_action,pred_value,_=model(seq,device)
        pred_action,_=model(seq,device)
        lower_weight=1.0/(pred_action.shape[1]-1)
        #lower_weight=0.0
        upper_weight=1.0
        policy_loss=0.0
        #value_loss=0.0
        for t in range(pred_action.shape[1]):
            if t<pred_action.shape[1]-1:
                w=lower_weight
            else:
                w=upper_weight
            policy_loss+=policy_loss_fn(pred_action[:,t,:],labels[:,t])
            #value_loss+=value_loss_fn(pred_value[:,t,:],values[:,t,:])
        #loss=policy_loss+value_loss
        #loss=value_loss
        loss=policy_loss
        loss.backward()
        optimizer.step()
        end_time=time.time()
        batch_dur=end_time-start_time
        if batch_idx%log_interval==0:
            with torch.no_grad():
                mask=labels>=4
                acc=torch.eq(pred_action[mask].argmax(1),labels[mask]).float().mean().item()*100.0
            log.info('[Epoch: ' + str(epoch) + '] ' + \
						'[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
						'[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
						'[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
						'[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')

def test(model,device,epoch,test_loader,log_interval=10):
    log.info("Evaluating test set...")
    model.eval()
    model.to(device)
    all_acc=[]
    for batch_idx,batch_data in enumerate(test_loader):
        start_time = time.time()
        seq,seq_lengths,labels,values=batch_data  
        #seq,seq_lengths,labels=batch_data
        seq=seq.to(device)
        labels=labels.to(device)
        #pred,_,_=model(seq,device)
        pred,_=model(seq,device)
        mask=labels>=4
        acc=torch.eq(pred[mask].argmax(1),labels[mask]).float().mean().item()*100.0
        all_acc.append(acc)
        end_time=time.time()
        batch_dur=end_time-start_time
        if batch_idx%log_interval==0:
            log.info('[Epoch: ' + str(epoch) + '] ' + \
                            '[Batch: ' + str(batch_idx) + ' of ' + str(len(test_loader)) + '] ' + \
                            '[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
                            '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
        
    log.info("Epoch "+str(epoch)+" TOTAL MEAN ACCURACY: "+str(np.mean(all_acc)))
    return np.mean(all_acc) 

if __name__=='__main__':
    n_rollouts=int(sys.argv[1])
    only_generate=int(sys.argv[2]) 
    lr=5e-4
    num_epochs=15
    num_nodes=7
    size=9 
    batch_size=256
    log_interval=100
    log.info("Importing dataset...")
    
    train_data=SupervisedTrajectoryDataset(num_train_nodes=[7,11],num_test_nodes=9,n_train_traj=1000000,n_test_traj=1000,train=True,strat_class=RolloutStrategy,n_rollouts=n_rollouts)
    test_data=SupervisedTrajectoryDataset(num_train_nodes=[7,11],num_test_nodes=9,n_train_traj=1000000,n_test_traj=1000,train=False,strat_class=RolloutStrategy,n_rollouts=n_rollouts)

    if only_generate==0:
        if torch.cuda.is_available():
            device=torch.device('cuda:0')
        else:
            device=torch.device('cpu')
        #device=torch.device('cpu')
        
        model=TinyRNN(input_size=train_data.obs_size,num_actions=train_data.num_actions)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        log.info("Start Training...")
        for i in range(num_epochs):
            epoch=i+1 
            train_loader=DataLoader(train_data,batch_size,shuffle=True,collate_fn=PadSequence())
            test_loader=DataLoader(test_data,batch_size,shuffle=True,collate_fn=PadSequence())
            train(model,device,optimizer,epoch,train_loader,log_interval=log_interval)
            test(model,device,epoch,test_loader,log_interval=log_interval)
            log.info("Saving Model...")
            torch.save(model.state_dict(),'models/6_dim_{}_rollout_model_epoch_{}.pt'.format(n_rollouts,epoch))



        



        
        




