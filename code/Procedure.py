import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)

    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri

        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()

    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            

def Val(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    valDict: dict = dataset.valDict

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    results = {
        'precision': np.zeros(len(world.topks)),
        'recall': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks))
    }

    try:
        with torch.no_grad():
            users = list(valDict.keys())

            if len(users) == 0:
                raise ValueError("Validation set is empty.")

            if u_batch_size > len(users) / 10:
                print(
                    f"val_u_batch_size is too big for this dataset, "
                    f"try a small one: {len(users) // 10}"
                )

            # Tính embedding toàn đồ thị một lần duy nhất
            all_users, all_items = Recmodel.computer()

            rating_list = []
            groundTrue_list = []

            total_batch = (
                len(users) + u_batch_size - 1
            ) // u_batch_size

            for batch_users in utils.minibatch(
                users, batch_size=u_batch_size
            ):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [valDict[u] for u in batch_users]

                batch_users_gpu = torch.as_tensor(
                    batch_users,
                    dtype=torch.long,
                    device=world.device
                )

                users_emb = all_users[batch_users_gpu]
                rating = Recmodel.f(
                    torch.matmul(users_emb, all_items.t())
                )

                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                if exclude_index:
                    rating[
                        exclude_index, exclude_items
                    ] = -(1 << 10)

                # Lấy top-K
                _, rating_K = torch.topk(
                    rating, k=max_K
                )

                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)

            assert total_batch == len(rating_list)

            X = list(zip(rating_list, groundTrue_list))

            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = [
                    test_one_batch(x) for x in X
                ]

            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']

            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))

            if world.tensorboard and w is not None:
                w.add_scalars(
                    f'Val/Recall@{world.topks}',
                    {
                        str(world.topks[i]): results['recall'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

                w.add_scalars(
                    f'Val/Precision@{world.topks}',
                    {
                        str(world.topks[i]): results['precision'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

                w.add_scalars(
                    f'Val/NDCG@{world.topks}',
                    {
                        str(world.topks[i]): results['ndcg'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

        print(results)
        return results

    finally:
        if multicore == 1:
            pool.close()
            pool.join()

    
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    testDict: dict = dataset.testDict

    # Chuyển mô hình sang chế độ đánh giá
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    results = {
        'precision': np.zeros(len(world.topks)),
        'recall': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks))
    }

    try:
        with torch.no_grad():
            users = list(testDict.keys())

            if len(users) == 0:
                raise ValueError("Test set is empty.")

            if u_batch_size > len(users) / 10:
                print(
                    f"test_u_batch_size is too big for this dataset, "
                    f"try a small one: {len(users) // 10}"
                )

            # Tính embedding toàn đồ thị một lần duy nhất
            all_users, all_items = Recmodel.computer()

            rating_list = []
            groundTrue_list = []

            total_batch = (
                len(users) + u_batch_size - 1
            ) // u_batch_size

            for batch_users in utils.minibatch(
                users, batch_size=u_batch_size
            ):
                # Lấy các sản phẩm người dùng đã tương tác
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]

                batch_users_gpu = torch.as_tensor(
                    batch_users,
                    dtype=torch.long,
                    device=world.device
                )

                # Tái sử dụng embedding đã tính
                users_emb = all_users[batch_users_gpu]

                rating = Recmodel.f(
                    torch.matmul(users_emb, all_items.t())
                )

                # Loại các sản phẩm đã tương tác khỏi danh sách đề xuất
                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                if exclude_index:
                    rating[
                        exclude_index, exclude_items
                    ] = -(1 << 10)

                # Lấy top-K sản phẩm
                _, rating_K = torch.topk(
                    rating, k=max_K
                )

                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)

            assert total_batch == len(rating_list)

            X = list(zip(rating_list, groundTrue_list))

            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = [
                    test_one_batch(x) for x in X
                ]

            # Tổng hợp các chỉ số đánh giá
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']

            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))

            # Ghi kết quả vào TensorBoard nếu được bật
            if world.tensorboard and w is not None:
                w.add_scalars(
                    f'Test/Recall@{world.topks}',
                    {
                        str(world.topks[i]): results['recall'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

                w.add_scalars(
                    f'Test/Precision@{world.topks}',
                    {
                        str(world.topks[i]): results['precision'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

                w.add_scalars(
                    f'Test/NDCG@{world.topks}',
                    {
                        str(world.topks[i]): results['ndcg'][i]
                        for i in range(len(world.topks))
                    },
                    epoch
                )

        print(results)
        return results

    finally:
        if multicore == 1:
            pool.close()
            pool.join()