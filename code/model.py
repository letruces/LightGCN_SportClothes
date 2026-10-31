import os
from pathlib import Path
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
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # Init embeeding 
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Init initial embedding
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        # Initialize Image Mapping  
        self.f_v = nn.Linear(768, self.latent_dim)
        nn.init.normal_(self.f_v.weight, std=0.1)
        nn.init.constant_(self.f_v.bias, 0.0)

        # Load file image vector
        CURRENT_FILE = Path(__file__).resolve()
        PROJECT_ROOT = CURRENT_FILE.parent.parent
        data_path = PROJECT_ROOT / 'data' / 'amazon-sport-and-outdoors'
        feat_path = os.path.join(data_path, 'visual_features.pt')
        mask_path = os.path.join(data_path, 'visual_has_image.pt')
        visual_feat = torch.load(feat_path) 
        has_image   = torch.load(mask_path)   

        # Register buffer
        self.register_buffer('visual_feat', visual_feat.float())
        self.register_buffer('has_image_mask', has_image.unsqueeze(1).float())

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lightvisualgcn is already to go(dropout:{self.config['dropout']})")

    def get_all_item_0_embeddings(self):
        """
        Formula:
        + Has image  : e_i^(0) = e_i^(ID) + W_v * Z_CLS + b_v
        + No image: e_i^(0) = e_i^(ID)
        """
        item_id_emb = self.embedding_item.weight         
        v_proj = self.f_v(self.visual_feat) 
        v_proj_masked = v_proj * self.has_image_mask 
        
        return item_id_emb + v_proj_masked

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
    
    def computer(self, items_emb=None):
        """
        Propagate methods for LightGCN (Optimized)
        """
        # Get intial embedding
        users_emb = self.embedding_user.weight
        if items_emb is None:
            items_emb = self.get_all_item_0_embeddings()
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
        return users, items
    
    def getUsersRating(self, users):
        # Tính điểm cho mỗi người dùng với mọi sản phẩm trong danh mục
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        items_emb_ego = self.get_all_item_0_embeddings()
        all_users, all_items = self.computer(items_emb=items_emb_ego)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = items_emb_ego[pos_items]
        neg_emb_ego = items_emb_ego[neg_items]
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