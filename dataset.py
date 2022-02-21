import torch.utils.data as DATA
from transformers import AdamW, AutoModel, AutoTokenizer
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
#import rdkit
#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
import torch
from sklearn.utils import shuffle
import torch
import random

class DDI2013Dataset(InMemoryDataset):
    def __init__(self, root='/tmp',
                 path='',
                 transform=None,
                 pre_transform=None,
                 max_len=128, model_name=''):

        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.path = path
        self.pass_list = []
        self.pass_smiles = set()
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
        self.NOTINDICT = 19
        # root is required for save preprocessed data, default is '/tmp'
        super(DDI2013Dataset, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_idx_split(self, data_size, train_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:])
        split_dict = {'train': train_idx, 'valid': val_idx}
        return split_dict

    def get_pos_z(self, smile1, i):
        # print(smile1)
        m1 = rdkit.Chem.MolFromSmiles(smile1)

        if m1 is None:
            self.pass_list.append(i)
            self.pass_smiles.add(smile1)
            return None, None

        if m1.GetNumAtoms() == 1:
            self.pass_list.append(i)
            if m1.GetNumAtoms() == 1:
                self.pass_smiles.add(smile1)
            return None, None
        m1 = Chem.AddHs(m1)

        ignore_flag1 = 0
        ignore1 = False

        while AllChem.EmbedMolecule(m1) == -1:
            print('retry')
            ignore_flag1 = ignore_flag1 + 1
            if ignore_flag1 >= 10:
                ignore1 = True
                break
        if ignore1:
            self.pass_list.append(i)
            self.pass_smiles.add(smile1)
            return None, None
        AllChem.MMFFOptimizeMolecule(m1)
        m1 = Chem.RemoveHs(m1)
        m1_con = m1.GetConformer(id=0)

        pos1 = []
        for j in range(m1.GetNumAtoms()):
            pos1.append(list(m1_con.GetAtomPosition(j)))
        np_pos1 = np.array(pos1)
        ten_pos1 = torch.Tensor(np_pos1)

        z1 = []
        for atom in m1.GetAtoms():
            if self.atomType.__contains__(atom.GetSymbol()):
                z = self.atomType[atom.GetSymbol()]
            else:
                z = self.NOTINDICT
            z1.append(z)

        z1 = np.array(z1)
        z1 = torch.tensor(z1)
        return ten_pos1, z1

    def process(self, root):
        df1 = pd.read_csv(root + 'raw/' + self.path)
        # df2 = pd.read_csv(root + 'raw/' + 'test.csv')
        data_list = []
        data_len = len(df1)

        smile_pos_dict = {}
        smile_z_dict = {}

        for i in range(data_len):
            print('Converting SMILES to 3Dgraph: {}/{}'.format(i + 1, data_len))
            sent = df1.loc[i, 'text']
            # sent = self.data.loc[idx, 'text']
            encoded_pair = self.tokenizer(sent,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=self.max_len,
                                          return_tensors='pt')
            token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
            attn_masks = encoded_pair['attention_mask'].squeeze(0)
            # binary tensor with "0" for padded values and "1" for the other values
            token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
            # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

            pos1 = []
            pos2 = []
            smile1 = df1.loc[i, 'smile1']
            smile2 = df1.loc[i, 'smile2']
            if self.pass_smiles.__contains__(smile1) or self.pass_smiles.__contains__(smile2):
                self.pass_list.append(i)
                continue

            if smile_pos_dict.__contains__(smile1):
                ten_pos1 = smile_pos_dict[smile1]
                z1 = smile_z_dict[smile1]
            else:
                ten_pos1, z1 = self.get_pos_z(smile1, i)
                if ten_pos1 == None:
                    continue
                else:
                    smile_pos_dict[smile1] = ten_pos1
                    smile_z_dict[smile1] = z1
            if smile_pos_dict.__contains__(smile2):
                ten_pos2 = smile_pos_dict[smile2]
                z2 = smile_z_dict[smile2]
            else:
                ten_pos2, z2 = self.get_pos_z(smile2, i)
                if ten_pos2 == None:
                    continue
                else:
                    smile_pos_dict[smile2] = ten_pos2
                    smile_z_dict[smile2] = z2

            label = df1.loc[i, 'label']
            label = np.array(label)
            label = torch.tensor(label)

            drug_pos1 = df1.loc[i, 'pos1']
            drug_pos2 = df1.loc[i, 'pos2']
            drug_pos1 = np.array(drug_pos1)
            drug_pos1 = torch.tensor(drug_pos1)

            drug_pos2 = np.array(drug_pos2)
            drug_pos2 = torch.tensor(drug_pos2)
            data = DATA(pos1=ten_pos1, z1=z1,
                        y=label,
                        pos2=ten_pos2, z2=z2,
                        token_ids=token_ids,
                        attn_masks=attn_masks,
                        token_type_ids=token_type_ids,
                        drug1_pos=drug_pos1,
                        drug2_pos=drug_pos2
                        )
            print(data)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        print(self.pass_list)
        print(len(self.pass_list))
        print(self.pass_smiles)
        print(len(self.pass_smiles))


class drugbankDataset(InMemoryDataset):
    def __init__(self, root='/tmp',
                 path='',
                 transform=None,
                 pre_transform=None):

        self.path = path
        self.pass_list = []
        self.pass_smiles = set()
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
        self.NOTINDICT = 19
        # root is required for save preprocessed data, default is '/tmp'
        super(drugbankDataset, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_idx_split(self, data_size, train_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:])
        split_dict = {'train': train_idx, 'valid': val_idx}
        return split_dict

    def get_pos_z(self, smile1, i):
        # print(smile1)
        m1 = rdkit.Chem.MolFromSmiles(smile1)

        if m1 is None:
            self.pass_list.append(i)
            self.pass_smiles.add(smile1)
            return None, None

        if m1.GetNumAtoms() == 1:
            self.pass_list.append(i)
            if m1.GetNumAtoms() == 1:
                self.pass_smiles.add(smile1)
            return None, None
        m1 = Chem.AddHs(m1)

        ignore_flag1 = 0
        ignore1 = False

        while AllChem.EmbedMolecule(m1) == -1:
            print('retry')
            ignore_flag1 = ignore_flag1 + 1
            if ignore_flag1 >= 10:
                ignore1 = True
                break
        if ignore1:
            self.pass_list.append(i)
            self.pass_smiles.add(smile1)
            return None, None
        AllChem.MMFFOptimizeMolecule(m1)
        m1 = Chem.RemoveHs(m1)
        m1_con = m1.GetConformer(id=0)

        pos1 = []
        for j in range(m1.GetNumAtoms()):
            pos1.append(list(m1_con.GetAtomPosition(j)))
        np_pos1 = np.array(pos1)
        ten_pos1 = torch.Tensor(np_pos1)

        z1 = []
        for atom in m1.GetAtoms():
            if self.atomType.__contains__(atom.GetSymbol()):
                z = self.atomType[atom.GetSymbol()]
            else:
                z = self.NOTINDICT
            z1.append(z)

        z1 = np.array(z1)
        z1 = torch.tensor(z1)
        return ten_pos1, z1

    def process(self, root):
        df1 = pd.read_csv(root + 'raw/' + self.path)
        # df2 = pd.read_csv(root + 'raw/' + 'test.csv')
        data_list = []
        data_len = len(df1)

        smile_pos_dict = {}
        smile_z_dict = {}

        h_to_t_dict = {}
        t_to_h_dict = {}
        id_set = set()
        id_smiles_dict = {}
        for i in range(df1.shape[0]):
            head = df1.loc[i, 'Drug1_ID']
            tail = df1.loc[i, 'Drug2_ID']
            head_smile = df1.loc[i, 'Drug1']
            tail_smile = df1.loc[i, 'Drug2']
            if h_to_t_dict.__contains__(head):
                h_to_t_dict[head].append(tail)
            else:
                h_to_t_dict[head] = []
                h_to_t_dict[head].append(tail)

            if t_to_h_dict.__contains__(tail):
                t_to_h_dict[tail].append(head)
            else:
                t_to_h_dict[tail] = []
                t_to_h_dict[tail].append(head)
            id_smiles_dict[head] = head_smile
            id_smiles_dict[tail] = tail_smile
            id_set.add(head)
            id_set.add(tail)


        for i in range(data_len):
            print('Converting SMILES to 3Dgraph: {}/{}'.format(i + 1, data_len))

            smile1 = df1.loc[i, 'Drug1']
            smile2 = df1.loc[i, 'Drug2']
            if self.pass_smiles.__contains__(smile1) or self.pass_smiles.__contains__(smile2):
                self.pass_list.append(i)
                continue

            if smile_pos_dict.__contains__(smile1):
                ten_pos1 = smile_pos_dict[smile1]
                z1 = smile_z_dict[smile1]
            else:
                ten_pos1, z1 = self.get_pos_z(smile1, i)
                if ten_pos1 == None:
                    continue
                else:
                    smile_pos_dict[smile1] = ten_pos1
                    smile_z_dict[smile1] = z1
            if smile_pos_dict.__contains__(smile2):
                ten_pos2 = smile_pos_dict[smile2]
                z2 = smile_z_dict[smile2]
            else:
                ten_pos2, z2 = self.get_pos_z(smile2, i)
                if ten_pos2 == None:
                    continue
                else:
                    smile_pos_dict[smile2] = ten_pos2
                    smile_z_dict[smile2] = z2

            label = torch.tensor(1)

            data = DATA(pos1=ten_pos1, z1=z1,
                        y=label,
                        pos2=ten_pos2, z2=z2,
                        )
            print(data)

            data_list.append(data)

            if random.random() > 0.5:  # 换尾
                head = df1.loc[i, 'Drug1_ID']
                tail_set = h_to_t_dict[head]
                pes_tail = random.sample(id_set - set(tail_set), 1)
                smile2 = id_smiles_dict[pes_tail[0]]
            else:
                tail = df1.loc[i, 'Drug2_ID']
                head_set = t_to_h_dict[tail]
                pes_head = random.sample(id_set - set(head_set), 1)
                smile1 = id_smiles_dict[pes_head[0]]

            if self.pass_smiles.__contains__(smile1) or self.pass_smiles.__contains__(smile2):
                self.pass_list.append(i)
                continue

            if smile_pos_dict.__contains__(smile1):
                ten_pos1 = smile_pos_dict[smile1]
                z1 = smile_z_dict[smile1]
            else:
                ten_pos1, z1 = self.get_pos_z(smile1, i)
                if ten_pos1 == None:
                    continue
                else:
                    smile_pos_dict[smile1] = ten_pos1
                    smile_z_dict[smile1] = z1
            if smile_pos_dict.__contains__(smile2):
                ten_pos2 = smile_pos_dict[smile2]
                z2 = smile_z_dict[smile2]
            else:
                ten_pos2, z2 = self.get_pos_z(smile2, i)
                if ten_pos2 == None:
                    continue
                else:
                    smile_pos_dict[smile2] = ten_pos2
                    smile_z_dict[smile2] = z2

            label = torch.tensor(0)

            data = DATA(pos1=ten_pos1, z1=z1,
                        y=label,
                        pos2=ten_pos2, z2=z2,
                        )
            print(data)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        print(self.pass_list)
        print(len(self.pass_list))
        print(self.pass_smiles)
        print(len(self.pass_smiles))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


