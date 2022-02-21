from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.nn.functional import selu
from SchNet import SchNet
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class myModel_text_graph_pos_cnn_sch(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_class=2, freeze_bert=False, max_len=128,
                 emb_dim=64,cutoff = 10.0,num_layers = 6,hidden_channels = 128,num_filters = 128,num_gaussians = 50,g_out_channels = 5):
        super(myModel_text_graph_pos_cnn_sch, self).__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.bert = AutoModel.from_pretrained(model_name, cache_dir='../cache', output_hidden_states=True,
                                              return_dict=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cnn = CNN()
        self.cutoff = cutoff
        self.num_layers =num_layers
        self.hidden_channels =hidden_channels
        self.num_filters =num_filters
        self.num_gaussians =num_gaussians
        self.model1 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)
        self.model2 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)

        self.fc_g_1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )
        self.fc_g_2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )

        self.cnn_g = CNN_g(in_channel=2, out_channel=num_class)

        self.emb = nn.Embedding(self.max_len + 1, self.emb_dim)

        self.fc_emb = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.emb_dim * 2, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, num_class, bias=True)
        )

        self.fc3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(3 * num_class, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, num_class, bias=True)
        )

    def forward(self, batch_data):
        outputs = self.bert(input_ids=batch_data.token_ids.view(-1, self.max_len),
                            token_type_ids=batch_data.token_type_ids.view(-1, self.max_len),
                            attention_mask=batch_data.attn_masks.view(-1, self.max_len))
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4, -5, -6]]),
                                  dim=-1).view(outputs.hidden_states[-1].shape[0], -1,
                                               outputs.hidden_states[-1].shape[1],
                                               outputs.hidden_states[-1].shape[-1])  # [bs, seq_len, hidden_dim*6]
        logits = self.cnn(hidden_states)
        batch_data.pos = batch_data.pos1
        batch_data.z = batch_data.z1
        batch_data.batch = batch_data.pos1_batch
        self.pred1 = self.model1(batch_data)
        batch_data.pos = batch_data.pos2
        batch_data.z = batch_data.z2
        batch_data.batch = batch_data.pos2_batch
        self.pred2 = self.model2(batch_data)
        self.pred1 = self.fc_g_1(self.pred1)
        self.pred2 = self.fc_g_2(self.pred2)
        self.pred1 = self.pred1.unsqueeze(1)
        self.pred2 = self.pred2.unsqueeze(1)
        self.pred = torch.cat((self.pred1, self.pred2), 1)
        self.pred = self.cnn_g(self.pred)

        # self.pred = (self.pred + 9*logits)/10.0

        drug1_pos = batch_data.drug1_pos
        drug2_pos = batch_data.drug2_pos
        drug1_pos[drug1_pos == -1] = self.max_len
        drug2_pos[drug2_pos == -1] = drug1_pos[drug2_pos == -1]
        self.emb1 = self.emb(drug1_pos)
        self.emb2 = self.emb(drug2_pos)
        self.emb_cat = torch.cat((self.emb1, self.emb2), 1)
        self.emb_cat = self.fc_emb(self.emb_cat)
        # self.emb_cat = F.softmax(self.emb_cat,dim=1)

        # self.pred = (19*self.pred + self.emb_cat)/20.0

        self.pred_total = torch.cat((logits, self.pred, self.emb_cat), 1)
        self.pred_total = self.fc3(self.pred_total)
        # if random.random() < 0.01:
        #     print(logits)
        #     print(self.pred)
        #     print(self.emb_cat)
        #     print(self.pred_total)
        return self.pred_total



class myModel_graph_sch_cnn(nn.Module):
    def __init__(self,num_class=2,cutoff = 10.0,num_layers = 6,hidden_channels = 128,
                 num_filters = 128,num_gaussians = 50,g_out_channels = 5):
        super(myModel_graph_sch_cnn, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.model1 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)
        self.model2 = SchNet(energy_and_force=False, cutoff=self.cutoff, num_layers=self.num_layers,
                             hidden_channels=self.hidden_channels, num_filters=self.num_filters, num_gaussians=self.num_gaussians,
                             out_channels=g_out_channels)

        self.fc1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )
        self.fc2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32, 32 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(32 * 2, 32, bias=True)
        )

        self.cnn = CNN_g(in_channel=2, out_channel=num_class)

    def forward(self, batch_data):
        batch_data.pos = batch_data.pos1
        batch_data.z = batch_data.z1
        batch_data.batch = batch_data.pos1_batch
        self.pred1 = self.model1(batch_data)
        batch_data.pos = batch_data.pos2
        batch_data.z = batch_data.z2
        batch_data.batch = batch_data.pos2_batch
        self.pred2 = self.model2(batch_data)

        self.pred1 = self.fc1(self.pred1)
        self.pred2 = self.fc2(self.pred2)
        self.pred1 = self.pred1.unsqueeze(1)
        self.pred2 = self.pred2.unsqueeze(1)
        self.pred = torch.cat((self.pred1, self.pred2), 1)
        self.pred = self.cnn(self.pred)
        return self.pred



class CNN(nn.Module):
    def __init__(self, in_channel=6, fc1_hid_dim=128 * 768, out_channel=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(1, 3), padding=[0, 1])
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=[0, 1])
        self.conv31 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=[0, 1])
        self.conv32 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=[0, 1])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=[0, 1])
        self.fc1 = nn.Linear(fc1_hid_dim, 64)
        self.fc2 = nn.Linear(64, out_channel)
        self.out_channel = out_channel
        self.Lrelu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
    def forward(self, x):

        x = self.Lrelu(self.bn1(self.conv1(x))) # 输入 batch size * hiddenLayers * max_len * embedding Length (bs*6*128*768) 输出 bs*64*128*768
        x = self.Lrelu(self.bn2(self.conv2(x)))  # 输出 bs*128*128*768
        res = x
        x = self.Lrelu(self.bn31(self.conv31(x)))  # 输出 bs*128*128*768
        x = self.Lrelu(self.bn32(self.conv32(x)))  # 输出 bs*128*128*768
        x = res + x
        x = self.Lrelu(self.bn4(self.conv4(x)))  # 输出 bs*256*128*768
        x = self.Lrelu(self.fc1(x.view(x.shape[0], x.shape[1], -1)))  # 输出 bs*64*64
        x = self.Lrelu(self.fc2(x))  # 输出 bs*64*out_channel
        x = F.adaptive_avg_pool2d(x, (1, self.out_channel)).squeeze(dim=-1).squeeze(1)  # 平均池化为 bs*out_channel

        return x




class CNN_g(nn.Module):
    def __init__(self, in_channel=2, fc1_hid_dim=256 * 32, out_channel=2):
        super(CNN_g, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv31 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv32 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(fc1_hid_dim, 64)
        self.fc2 = nn.Linear(64, out_channel)
        self.Lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.Lrelu(self.conv1(x))  # batchsize *2 * 32 变为 batchsize *64 * 32
        x = self.Lrelu(self.conv2(x))  # batchsize *128 * 32
        res = x
        x = self.Lrelu(self.conv31(x))  # batchsize *128 * 32
        x = self.Lrelu(self.conv32(x))  # batchsize *128 * 32
        x = res + x  # batchsize *128 * 32
        x = self.Lrelu(self.conv4(x))  # batchsize *256 * 32
        x = self.Lrelu(self.fc1(x.view(x.shape[0], -1)))  # batchsize * 64
        x = self.fc2(x)  # batchsize * out_channel 输出通道数代表预测的类别数量 根据任务的分类类别来确定  ddi2013里面是5 drugbank里面是2

        return x




import torch.nn.functional as F


class myModel_text_cnn(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_class=2, freeze_bert=False,
                 max_len=128):  # , freeze_bert=False   ,model_name):
        super(myModel_text_cnn, self).__init__()
        self.max_len = max_len
        self.bert = AutoModel.from_pretrained(model_name, cache_dir='../cache', output_hidden_states=True,
                                              return_dict=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size * 6, num_class, bias=False)
        )
        self.cnn = CNN()

    def forward(self, batch_data):
        outputs = self.bert(input_ids=batch_data.token_ids.view(-1, self.max_len),
                            token_type_ids=batch_data.token_type_ids.view(-1, self.max_len),
                            attention_mask=batch_data.attn_masks.view(-1, self.max_len))
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4, -5, -6]]),
                                  dim=-1).view(outputs.hidden_states[-1].shape[0], -1,
                                               outputs.hidden_states[-1].shape[1],
                                               outputs.hidden_states[-1].shape[-1])  # [bs, seq_len, hidden_dim*6]
        self.pred = self.cnn(hidden_states)
        return self.pred




class myModel_text_pos_cnn(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_class=2, freeze_bert=False, max_len=128,
                 emb_dim=64):  # , freeze_bert=False   ,model_name):
        super(myModel_text_pos_cnn, self).__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.bert = AutoModel.from_pretrained(model_name, cache_dir='../cache', output_hidden_states=True,
                                              return_dict=True)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size * 6, num_class, bias=False)
        )

        self.emb = nn.Embedding(self.max_len + 1, self.emb_dim)

        self.fc_emb = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.emb_dim * 2, 32 * 2, bias=False),
            nn.PReLU(),
            nn.Linear(32 * 2, num_class, bias=False)
        )
        self.cnn = CNN()

    def forward(self, batch_data):
        outputs = self.bert(input_ids=batch_data.token_ids.view(-1, self.max_len),
                            token_type_ids=batch_data.token_type_ids.view(-1, self.max_len),
                            attention_mask=batch_data.attn_masks.view(-1, self.max_len))
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4, -5, -6]]),
                                  dim=-1).view(outputs.hidden_states[-1].shape[0], -1,
                                               outputs.hidden_states[-1].shape[1],
                                               outputs.hidden_states[-1].shape[-1])  # [bs, seq_len, hidden_dim*6]
        logits = self.cnn(hidden_states)

        drug1_pos = batch_data.drug1_pos
        drug2_pos = batch_data.drug2_pos
        drug1_pos[drug1_pos == -1] = self.max_len
        drug2_pos[drug2_pos == -1] = drug1_pos[drug2_pos == -1]
        self.emb1 = self.emb(drug1_pos)
        self.emb2 = self.emb(drug2_pos)
        self.emb_cat = torch.cat((self.emb1, self.emb2), 1)
        self.emb_cat = self.fc_emb(self.emb_cat)

        self.pred = (logits + 0.1 * self.emb_cat) / 1.1
        return self.pred


class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss
