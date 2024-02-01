import argparse

datasets_args = {
    "austin": {
        "type_num": [7355]
    },
    "sf_la": {
        "type_num": [6393]
    },
    "nyc": {
        "type_num": [13692]
    },
}


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--true_1_weight", type=float, default=3.0)
    parser.add_argument("--true_2_weight", type=float, default=1.0)
    parser.add_argument("--true_3_weight", type=float, default=0.2)
    parser.add_argument("--true_4_weight", type=float, default=0.1)
    parser.add_argument("--neg_weight", type=float, default=0.1)
    parser.add_argument("--file_name", type=str, default='default_res.txt')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--hidden_dim', type=int, default=128)  # 128
    parser.add_argument('--num_pos_samples', type=int, default=40)
    parser.add_argument('--num_pos_samples_ulu', type=int, default=4)
    parser.add_argument('--neg_sample_size', type=int, default=4)
    parser.add_argument('--walk_len', type=int, default=5)
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--walk_len_ulu', type=int, default=5)
    parser.add_argument('--num_walks_ulu', type=int, default=100)
    # from GraphMAE
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat_drop", type=float, default=0.1,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.2,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default='batchnorm')

    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--feat_mask_rate", type=str, default="0.5,0.005,0.8",
                        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                        If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                        If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""")
    parser.add_argument("--replace_rate", type=float, default=0.3,
                        help="The replace rate. The ratio of nodes that is replaced by random nodes.")
    parser.add_argument("--leave_unchanged", type=float, default=0.3,
                        help="The ratio of nodes left unchanged (no mask), but is asked to reconstruct.")

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=1, help="pow index for sce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")

    parser.add_argument('--dataset', type=str, default="austin")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mae_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)  # 0.05
    parser.add_argument('--eva_wd', type=float, default=0, help="weight decay")

    # The parameters of learning process
    parser.add_argument('--patience', type=int,
                        default=40)  # 5. we can change the value. this might impact the performance.
    parser.add_argument('--l2_coef', type=float, default=0)

    # The parameters of metapath2vec
    parser.add_argument("--use_mp2vec_feat_pred", action="store_true", default=False, help="Set to True to use the mp2vec feature regularization.")
    parser.add_argument("--mps_lr", type=float, default=0.001, help="mp2vec learning rate")
    parser.add_argument('--mps_embedding_dim', type=int, default=128)
    parser.add_argument('--mps_walk_length', type=int, default=10)
    parser.add_argument('--mps_context_size', type=int, default=5)
    parser.add_argument('--mps_walks_per_node', type=int, default=1)
    parser.add_argument('--mps_num_negative_samples', type=int, default=3)
    parser.add_argument('--mps_batch_size', type=int, default=256)
    parser.add_argument('--mps_epoch', type=int, default=20)
    parser.add_argument('--mp2vec_feat_pred_loss_weight', type=float, default=0.1)
    parser.add_argument("--mp2vec_feat_alpha_l", type=float, default=1, help="pow index for sce loss in edge reconstruction")
    parser.add_argument("--mp2vec_feat_drop", type=float, default=.1, help="input feature dropout")

    # read config
    parser.add_argument("--use_cfg", action="store_true", help="Set to True to read config file")

    # meta-path edge reconstruction
    parser.add_argument("--use_mp_edge_recon", action="store_true", default=True,
                        help="Set to True to use the meta-path edge reconstruction.")
    parser.add_argument('--mp_edge_recon_loss_weight', type=float, default=0.5)
    parser.add_argument("--mp_edge_mask_rate", type=str, default="0.0",
                        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                        If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                        If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""")
    parser.add_argument("--mp_edge_alpha_l", type=float, default=3, help="pow index for sce loss in edge reconstruction")

    parser.add_argument("--task", type=str, default="classification", choices=["classification", "clustering"])

    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args
