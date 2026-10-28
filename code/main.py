import world
import utils
from world import cprint
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED: ", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
print("DEBUG 1: Before model.to()", flush=True)
Recmodel = Recmodel.to(world.device)
print(f"DEBUG 2: After model.to(), device={world.device}", flush=True)

print("DEBUG 3: Before BPRLoss", flush=True)
bpr = utils.BPRLoss(Recmodel, world.config)
print("DEBUG 4: After BPRLoss", flush=True)

print("DEBUG 5: Before getFileName", flush=True)    
weight_file = utils.getFileName()
print(f"DEBUG 6: Weight file = {weight_file}", flush=True)

print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1
print("DEBUG: Passed Neg_k", flush=True)

print("DEBUG: Before TensorBoard initialization", flush=True)
print("world.tensorboard =", world.tensorboard, flush=True)
# init tensorboard
if world.tensorboard:
    print("DEBUG: Initializing TensorBoard", flush=True)
    
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

# Lưu checkpoint có kết quả validation tốt nhất
best_weight_file = weight_file + ".best"
best_val_recall = -1.0
best_epoch = -1

# Chọn Recall@K làm tiêu chí chọn mô hình
target_k = 20
if target_k not in world.topks:
    raise ValueError(
        f"target_k={target_k} not in world.topks={world.topks}"
    )

recall_index = world.topks.index(target_k)

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # 1. Huấn luyện mô hình bằng tập train
        print('===========Start epoch================')
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

        if epoch % 10 == 0:
            # 2. Đánh giá mô hình bằng tập validation
            cprint("[VAL]")
            val_result = Procedure.Val(dataset, Recmodel, epoch, w, world.config['multicore'])

            # 3. Lấy Recall@K để chọn mô hình tốt nhất
            current_recall = float(val_result['recall'][recall_index])
            if current_recall > best_val_recall:
                best_val_recall = current_recall
                best_epoch = epoch

                torch.save(Recmodel.state_dict(), best_weight_file)

                print(
                    f"Best model: epoch={epoch + 1}, "
                    f"Recall@{target_k}={best_val_recall:.4f}"
                )

    # 4. Tải checkpoint tốt nhất để đánh giá cuối cùng
    if best_epoch >= 0:
        Recmodel.load_state_dict(torch.load(best_weight_file, map_location=world.device))

        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best validation Recall@{target_k}: {best_val_recall:.4f}")

        # 5. Đánh giá cuối cùng bằng tập test
        print("=== FINAL TEST EVALUATION ===")
        Procedure.Test(dataset, Recmodel, best_epoch, w, world.config['multicore'])
    else:
        print("No best model was saved. Please check validation results.")
        
finally:
    if world.tensorboard:
        w.close()

