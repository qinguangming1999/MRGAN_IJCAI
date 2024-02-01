import datetime
import warnings

import torch

from models.edcoder import PreModel
from utils.load_data import (load_data, neg_sample)
from utils.utils import get_roc_score, set_random_seed
from utils.params import build_args

warnings.filterwarnings('ignore')


args = build_args()
if torch.cuda.is_available():
    args.device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device("cpu")

print(args)

# random seed
set_random_seed(args.seed)

print("loading_data")
feats, mps, val_edge, val_edge_false, test_edge, test_edge_false, train_edge_f, train_edge, train_edge_ulu, train_edge_ullu = \
    load_data(args.type_num ,args.num_pos_samples, args.walk_len,args.num_walks, args.num_pos_samples_ulu,args.walk_len_ulu,args.num_walks_ulu)


train_edge_f_index = torch.Tensor(train_edge_f).to(args.device).long()
train_edge_index = torch.Tensor(train_edge).to(args.device).long()
train_edge_ulu_index = torch.Tensor(train_edge_ulu).to(args.device).long()
train_edge_ullu_index = torch.Tensor(train_edge_ullu).to(args.device).long()

feats_dim_list = [i.shape[1] for i in feats]

num_mp = int(len(mps))
print("Dataset: ", args.dataset)
print("The number of meta-paths: ", num_mp)


# model
focused_feature_dim = feats_dim_list[0]
model = PreModel(args, num_mp, focused_feature_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
# scheduler
if args.scheduler:
    print("--- Use schedular ---")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
else:
    scheduler = None

model.to(args.device)
feats = [feat.to(args.device) for feat in feats]
mps = [mp.to(args.device) for mp in mps]


cnt_wait = 0
best = 0
best_t = 0
best_test_roc = 0
best_test_ap = 0
starttime = datetime.datetime.now()
best_model_state_dict = None
for epoch in range(args.mae_epochs):#args.mae_epochs
    train_edge_false_index = neg_sample(args.neg_sample_size, args.type_num[0])
    train_edge_false_index = torch.Tensor(train_edge_false_index).to(args.device).long()
    model.train()
    optimizer.zero_grad()
    loss, loss_item = model(feats, mps, train_edge_false_index, train_edge_f_index[:,epoch%args.num_pos_samples],
                            train_edge_index[:,epoch%args.num_pos_samples],train_edge_ulu_index[:,epoch%args.num_pos_samples_ulu],
                            train_edge_ullu_index[:,epoch%args.num_pos_samples_ulu],epoch=epoch)
    print(f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}")
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    with torch.no_grad():
        model.eval()
        embeds = model.get_embeds(feats, mps)
        val_roc_curr, val_ap_curr = get_roc_score(val_edge, val_edge_false, embeds)
        print("val_roc : {:.4f}, val_ap : {:.4f}".format(val_roc_curr, val_ap_curr,))
        if val_ap_curr > best:
            best = val_ap_curr
            best_t = epoch
            cnt_wait = 0
            test_roc_curr, test_ap_curr = get_roc_score(test_edge, test_edge_false, embeds)
            print("test_roc_curr : {:.4f}, test_ap_curr : {:.4f}".format(test_roc_curr, test_ap_curr))
            best_test_roc = test_roc_curr
            best_test_ap = test_ap_curr
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
print('The best epoch is: ', best_t)

print("best_test_roc : {:.4f}, best_test_ap : {:.4f}".format(best_test_roc, best_test_ap))

endtime = datetime.datetime.now()
time = (endtime - starttime).seconds
print("Total time: ", time, "s")




