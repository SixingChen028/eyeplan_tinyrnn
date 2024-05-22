from utils import log
from model import TinyRNN
from dataset import * 
import torch 
import torch.nn as nn 
import time 
from torch.utils.data import DataLoader,random_split 
import torch.optim as optim

def train(model,device,optimizer,epoch,train_loader,log_interval=10):
    log.info("Starting epoch {}".format(epoch))
    model.train()
    model.to(device)
    loss_fn=nn.CrossEntropyLoss(ignore_index=-1)
    for batch_idx,batch_data in enumerate(train_loader):
        start_time = time.time()
        seq,seq_lengths,labels=batch_data  
        labels=labels.to(device)
        seq=seq.to(device)
        optimizer.zero_grad() 
        pred,_=model(seq,device)
        loss=0.0
        for t in range(pred.shape[1]):
            loss+=loss_fn(pred[:,t,:],labels[:,t])
        loss.backward()
        optimizer.step()
        end_time=time.time()
        batch_dur=end_time-start_time
        if batch_idx%log_interval==0:
            with torch.no_grad():
                mask=labels>-1
                acc=torch.eq(pred[mask].argmax(1),labels[mask]).float().mean().item()*100.0
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
        seq,seq_lengths,labels=batch_data 
        seq=seq.to(device)
        labels=labels.to(device)
        pred,_=model(seq,device)
        mask=labels>-1
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
    lr=5e-4
    num_epochs=1
    num_nodes=7
    batch_size=256
    hidden_size=8
    log_interval=100
    log.info("Importing dataset...")
    
    dataset=SupervisedTrajectoryDataset(num_node=num_nodes)
    train_data,test_data=random_split(dataset,[0.95,0.05])
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    #device=torch.device('cpu')

    dummy_env=DecisionTreeEnv(num_nodes)
    model=TinyRNN(input_size=dummy_env.observation_space.shape[0],hidden_size=hidden_size,num_actions=dummy_env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log.info("Start Training...")
    for i in range(num_epochs):
        epoch=i+1 
        train_loader=DataLoader(train_data,batch_size,shuffle=True,collate_fn=PadSequence())
        test_loader=DataLoader(test_data,batch_size,shuffle=True,collate_fn=PadSequence())
        train(model,device,optimizer,epoch,train_loader,log_interval=log_interval)
        test(model,device,epoch,test_loader,log_interval=log_interval)
    log.info("Finish training")
    torch.save(model.state_dict(),'8_dim_model.pt')



    



        
        




