import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def valDict(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    def __init__(self,config = world.config,path="../data/amazon-sport-and-outdoors"):
        # train or val or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, 'val': 1, 'test': 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        val_file = path + '/val.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        valUniqueUsers, valItem, valUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self._trainDataSize = 0
        self._valDataSize = 0
        self._testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self._trainDataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self._valDataSize += len(items)
        self.valUniqueUsers = np.array(valUniqueUsers)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self._testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.valDataSize} interactions for validating")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.n_user} users")
        print(f"{self.m_items} items")
        total_interactions = self.trainDataSize + self.valDataSize + self.testDataSize
        sparsity = total_interactions / (self.n_users * self.m_items) if (self.n_users * self.m_items) > 0 else 0
        print(f"{world.dataset} Sparsity : {sparsity}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__valDict = self.__build_val()
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self._trainDataSize

    @property
    def valDataSize(self):
        return self._valDataSize

    @property
    def testDataSize(self):
        return self._testDataSize

    @property
    def valDict(self):
        return self.__valDict

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix...")

        if self.Graph is None:
            try:
                norm_adj = sp.load_npz(
                    self.path + '/s_pre_adj_mat.npz'
                )
                print("successfully loaded...")

            except Exception as e:
                print("generating adjacency matrix...")
                s = time()

                # 1. Lấy tọa độ tương tác User-Item
                user_dim = self.n_users
                item_dim = self.m_items
                n_nodes = user_dim + item_dim

                R_coo = self.UserItemNet.tocoo()

                users = R_coo.row
                items = R_coo.col + user_dim

                # 2. Xây dựng ma trận kề đối xứng
                rows = np.concatenate([users, items])
                cols = np.concatenate([items, users])

                data = np.ones(len(rows), dtype=np.float32)

                adj_mat = sp.coo_matrix(
                    (data, (rows, cols)),
                    shape=(n_nodes, n_nodes),
                    dtype=np.float32
                )

                # 3. Tính bậc của các nút
                rowsum = np.asarray(
                    adj_mat.sum(axis=1)
                ).flatten()

                # 4. Tính D^(-1/2) an toàn
                d_inv = np.zeros_like(
                    rowsum, dtype=np.float32
                )

                mask = rowsum > 0
                d_inv[mask] = np.power(
                    rowsum[mask], -0.5
                )

                # 5. Chuẩn hóa từng phần tử khác 0
                norm_data = (
                    data * d_inv[rows] * d_inv[cols]
                )

                # 6. Chuyển sang CSR
                norm_adj = sp.csr_matrix(
                    (norm_data, (rows, cols)),
                    shape=(n_nodes, n_nodes),
                    dtype=np.float32
                )

                norm_adj.eliminate_zeros()

                end = time()
                print(
                    f"costing {end - s:.2f}s, "
                    "saved norm_mat..."
                )

                sp.save_npz(
                    self.path + '/s_pre_adj_mat.npz',
                    norm_adj
                )

            # 7. Chuyển ma trận sang Tensor trên thiết bị
            if self.split:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(
                    norm_adj
                )
                self.Graph = self.Graph.coalesce().to(
                    world.device
                )
                print("don't split the matrix")

        return self.Graph

    def __build_val(self):
        """
        return:
            dict: {user: [items]}
        """
        val_data = {}
        for i, item in enumerate(self.valItem):
            user = self.valUser[i]
            if val_data.get(user):
                val_data[user].append(item)
            else:
                val_data[user] = [item]
        return val_data
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
