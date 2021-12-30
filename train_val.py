from utils import *
from model import *
from dataset import *
from torch_geometric.loader import DataLoader
import time
import datetime
from tensorboardX import SummaryWriter

def train_eval(model, optimizer, train_loader, val_loader,test_loader, epochs=2 , log_path = 'default'):
    x = 0.005
    criterion = FocalLoss(alpha_t=[x, 0.19, 0.12, 0.6, 0.12], gamma=2)
    
    writer1 = SummaryWriter('../runs/' + log_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime
    bestf1 = 0
    for epoch in range(epochs):
        if epoch %5 == 0:
            x = x+0.0005
            criterion = FocalLoss(alpha_t=[x, 0.19, 0.12, 0.6, 0.12], gamma=2)
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        print('Epoch', epoch)
        model.train()
        for i, batch_data in enumerate(tqdm(train_loader)):
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            loss = criterion(logits, batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        print('-----Vailding-----')
        label, pred = get_label_pred_ddi2013(model, val_loader)
        f1 = sk_print_p_r(label, pred)
        print('-----Testing-----')
        label, pred = get_label_pred_ddi2013(model, test_loader)
        f1 = sk_print_p_r(label, pred)
        if f1 > bestf1:
            bestf1 = f1
            path = '../model/'+log_path+'.pt'
            torch.save(model.state_dict(), path)
        scheduler.step()
        writer1.add_scalar('macro_f1', f1, global_step=epoch, walltime=None)


def train_eval_drugbank(model, optimizer, train_loader, val_loader, epochs=2, log_path='./'):
    criterion = nn.CrossEntropyLoss()
    writer1 = SummaryWriter('../runs/' + log_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime
    bestacc = 0
    for epoch in range(epochs):
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        print('Epoch', epoch)
        model.train()
        for i, batch_data in enumerate(tqdm(train_loader)):
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            loss = criterion(logits, batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        # print('-----Vailding-----')
        label, pred,prob = get_label_pred_prob(model, val_loader)
        print('roc_auc:', sklearn.metrics.roc_auc_score(label, prob))
        area = sklearn.metrics.average_precision_score(label, prob)
        print('pr_auc:', area)
        acc = sklearn.metrics.accuracy_score(label, pred)
        print('acc:', acc)
        if acc > bestacc:
            bestacc = acc
            path = '../model/' + log_path + '.pt'
            torch.save(model.state_dict(), path)
        scheduler.step()
        writer1.add_scalar('acc', acc, global_step=epoch, walltime=None)

