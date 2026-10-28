import world
import torch
import dataloader
from dataloader import BasicDataset
from torch import nn
import numpy as np

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        # Số chiều của vector embedding mỗi người dùng và sản phẩm
        self.latent_dim = self.config['latent_dim_rec']
        # Số lớp lan truyền trên đồ thị
        self.n_layers = self.config['lightGCN_n_layers']
        # Xác suất giữ lại phần tử khi thực hiện dropout trên đồ thị trong quá trình huấn luyện
        self.keep_prob = self.config['keep_prob']
        # Cho biết đồ thị có được chia thành các phần nhỏ để xử lý hay không
        self.A_split = self.config['A_split']

        # Tạo embedding cho người dùng
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # Tạo embedding cho sản phẩm
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Khởi tạo giá trị embedding ban đầu
        if self.config['pretrain'] == 0:
            # Mô hình khởi tạo embedding bằng các giá trị ngẫu nhiên lấy từ phân phối chuẩn có độ lệch chuẩn 0,1
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        Propagate methods for LightGCN (Optimized)
        """
        print("========start computer=========",flush=True)
        # 1. Lấy embedding ban đầu
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        if self.config.get('dropout', False) and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph

        light_out = all_emb
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = [torch.sparse.mm(g, all_emb) for g in g_droped]
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            
            light_out = light_out + all_emb

        light_out = light_out / (self.n_layers + 1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        print("============done computer===========")
        return users, items
    
    def getUsersRating(self, users):
        # Tính điểm cho mỗi người dùng với mọi sản phẩm trong danh mục
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users, pos, neg)

        reg_loss = (1/2) * (
            (userEmb0 ** 2).sum() + 
            (posEmb0 ** 2).sum()  + 
            (negEmb0 ** 2).sum()
        ) / float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=-1)

        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # Tính điểm tương tác giữa từng user và item tương ứng
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma