import untangle
import pandas as pd
import numpy as np
import os
import re
import random
import torch

import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


def get_file(paths, drugbank):  # 获取文件路径
    i = 0
    for path in paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                # root = etree.parse(xml_path, parser=etree.XMLParser())

                obj = untangle.parse(str(path + file))
                obj = obj.document.children
                # print(obj)
                for sen in obj:
                    # print(sen['id'],sen['text'])
                    drug_dict = {}
                    try:
                        sen.entity
                    except:
                        count = 0
                    else:
                        for drug in sen.entity:
                            drug_dict[drug["id"]] = drug["text"]
                    try:
                        sen.pair
                    except:
                        count = 0
                    else:
                        for pair in sen.pair:
                            if pair['ddi'] == 'false' and random.random() > 0.1:
                                continue

                            text = sen['text']
                            text = text.replace(drug_dict[pair["e1"]], "DRUG1")
                            text = text.replace(drug_dict[pair["e2"]], "DRUG2")
                            for drug in drug_dict:
                                text = text.replace(drug_dict[drug], "DRUGOTHER")
                            drugbank.loc[i, "text"] = text
                            drugbank.loc[i, "drug1"] = drug_dict[pair["e1"]]
                            drugbank.loc[i, "drug2"] = drug_dict[pair["e2"]]
                            drugbank.loc[i, "ddi"] = pair['ddi']
                            if pair['ddi'] == 'true':
                                drugbank.loc[i, "type"] = pair['type']
                            i = i + 1
                # break
                return drugbank


def process_data(paths):
    drugbank = pd.DataFrame(
        columns=["text", "drug1", "drug2", "ddi", "type"])
    drugbank = get_file(paths, drugbank)
    drugbank.fillna('Neg', inplace=True)
    label_dict = {"Neg": 0,
                  'advise': 1,
                  'effect': 2,
                  'int': 3,
                  'mechanism': 4}
    drugbank['label'] = drugbank['type']
    for i in range(drugbank.shape[0]):
        drugbank['label'][i] = label_dict[drugbank['type'][i]]

    drugbank.to_csv("./train_sample10.csv")


def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data_df = sklearn.utils.shuffle(data_df, random_state=random_state)
    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


from matplotlib.colors import LinearSegmentedColormap


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, title_add=0,
                          polt_lim=0):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    colors = []
    for l in np.linspace(0, 1, 100):
        colors.append((30. / 255, 136. / 255, 229. / 255, l))
    transparent_blue = LinearSegmentedColormap.from_list("transparent_blue", colors)

    if title_add < polt_lim:
        return
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm.T)

    plt.imshow(cm.T, interpolation='nearest', cmap=cmap)
    if normalize:
        plt.title('Normalized ' + title)
    else:
        plt.title(title)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if float(num) > thresh else "black")
    plt.tight_layout(pad=1.3)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('save_img_' + str(title_add) + '.jpg')
    plt.close()


def plot_cm_and_get_label_pred(mymodel, val_loader, normalize=False, best_f1=0):
    mymodel.to(device)
    mymodel.eval()
    conf_matrix = torch.zeros(5, 5)
    types = ['Neg', 'Adv', 'Eff', 'Int', 'Mec']
    ma_f1 = 0
    pred, label = [], []
    for i, batch in enumerate(tqdm(val_loader)):
        # batch = tuple(t.to(device) for t in batch)
        batch = batch.to(device)
        # out = mymodel(batch[0], batch[1], batch[2])
        out = mymodel(batch)
        # conf_matrix = confusion_matrix(np.argmax(out.detach().cpu().numpy(),axis=1), labels=batch[-1], conf_matrix=conf_matrix)
        conf_matrix = confusion_matrix(np.argmax(out.detach().cpu().numpy(), axis=1), labels=batch.y,
                                       conf_matrix=conf_matrix)
        pred.extend(np.argmax(out.detach().cpu().numpy(), axis=1).flatten())
        # label.extend(batch[-1].detach().cpu().numpy().flatten())
        label.extend(batch.y.detach().cpu().numpy().flatten())
    ma_p = precision_score(np.array(label), np.array(pred), average='macro')
    ma_r = recall_score(np.array(label), np.array(pred), average='macro')
    ma_f1 = (2 * ma_p * ma_r) / (ma_p + ma_r)
    plot_confusion_matrix(conf_matrix.numpy(), classes=types, normalize=normalize, title_add=ma_f1, polt_lim=best_f1)
    return conf_matrix, np.array(label), np.array(pred)


def get_label_pred_ddi2013(mymodel, val_loader):
    mymodel.to(device)
    mymodel.eval()
    pred, label = [], []
    for i, batch in enumerate(val_loader):
        batch = batch.to(device)
        out = mymodel(batch)
        pred.extend(np.argmax(out.detach().cpu().numpy(), axis=1).flatten())
        label.extend(batch.y.detach().cpu().numpy().flatten())
    return np.array(label), np.array(pred)


def calculate_accuracy(cm):
    return torch.trace(cm) / torch.sum(cm)


def calculate_recall(cm, idx):
    return cm[idx][idx] / cm.sum(axis=0)[idx]


def calculate_precision(cm, idx):
    return cm[idx][idx] / cm.sum(axis=1)[idx]


def print_p_r(cm):
    precision_sum = 0
    recall_sum = 0
    precision = 0
    recall = 0
    f1_sum = 0
    print('accuracy:', calculate_accuracy(cm))
    for i in range(cm.shape[0]):
        precision = calculate_precision(cm, i)
        recall = calculate_recall(cm, i)
        precision_sum += precision
        recall_sum += recall
        print('label:', i, '  precision:', precision)
        print('label:', i, '  recall:', recall)
        f1_sum += (precision * recall * 2) / (precision + recall)
    macro_precision = precision_sum / cm.shape[0]
    macro_recall = recall_sum / cm.shape[0]
    print('Macro_precision:', macro_precision)
    print('Macro_recall:', macro_recall)
    print('Macro_F1:', f1_sum / cm.shape[0])


def sk_print_p_r(label, pred):
    ma_p = precision_score(np.array(label), np.array(pred), average='macro')
    ma_r = recall_score(np.array(label), np.array(pred), average='macro')
    ma_f1 = (2 * ma_p * ma_r) / (ma_p + ma_r)
    print('macro f1', ma_f1,
          'macro p', ma_p,
          'macro r', ma_r)
    print('micro f1', f1_score(label, pred, average='micro'))
    #print(classification_report(label,pred,digits=4))
    return ma_f1


def get_label_pred_prob(mymodel, val_loader):
    mymodel.to(device)
    mymodel.eval()
    ma_f1 = 0
    pred, label = [], []
    prob = []
    for i, batch in enumerate(tqdm(val_loader)):
        batch = batch.to(device)
        out = mymodel(batch)
        prob.extend(torch.sigmoid(out)[:, 1].detach().cpu().numpy().flatten())
        pred.extend(np.argmax(out.detach().cpu().numpy(), axis=1).flatten())
        label.extend(batch.y.detach().cpu().numpy().flatten())
    return np.array(label), np.array(pred), np.array(prob)
