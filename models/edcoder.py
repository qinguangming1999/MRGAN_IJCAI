from itertools import chain
from functools import partial
import torch
import torch.nn as nn
import dgl

from utils.utils import create_norm
from models.han import HAN


class PreModel(nn.Module):
    def __init__(
            self,
            args,
            num_metapath: int,
            focused_feature_dim: int
    ):
        super(PreModel, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = self.focused_feature_dim
        self.true_1_weight = args.true_1_weight
        self.true_2_weight = args.true_2_weight
        self.true_3_weight = args.true_3_weight
        self.true_4_weight = args.true_4_weight
        self.neg_weight = args.neg_weight
        self.neg_sample_size = args.neg_sample_size
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # num head: encoder
        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        # num head: decoder
        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        # encoder
        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        # decoder
        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )

        self.__cache_gs = None

        self.use_mp_edge_recon = args.use_mp_edge_recon
        self.mp_edge_recon_loss_weight = args.mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.mp_edge_alpha_l = args.mp_edge_alpha_l
        #self.mp_edge_recon_loss = self.setup_loss_fn(self.loss_fn, self.mp_edge_alpha_l)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)


    @property
    def output_hidden_dim(self):
        return self.hidden_dim

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def forward(self, feats, mps, train_edge_false_index, train_edge_f_index, train_edge_index,train_edge_ulu_index,train_edge_ullu_index,**kwargs):

        origin_feat = feats[0]

        loss = 0.
        # mp based edge reconstruction
        if self.use_mp_edge_recon:
            edge_recon_loss = self.mask_mp_edge_reconstruction(origin_feat, mps, train_edge_false_index,
                                                               train_edge_f_index, train_edge_index,train_edge_ulu_index,train_edge_ullu_index,kwargs.get("epoch", None))
            loss += self.mp_edge_recon_loss_weight * edge_recon_loss

        return loss, loss.item()

    def affinity(self, inputs1, inputs2):
        # element-wise production
        # 1-D tensor of shape (batch_size, )
        result = torch.sum(inputs1 * inputs2, dim=1)
        return result
    def neg_cost(self, inputs1, neg_samples):
        # neg sample size: (batch_size, num_neg_samples, input_dim)
        # (batch_size, 1, input_dim)
        inputs1_reshaped = torch.unsqueeze(inputs1, dim=1)
        # tensor of shape (batch_size, 1, num_neg_samples)
        neg_aff = torch.matmul(inputs1_reshaped, torch.transpose(neg_samples, 1, 2))
        # squeeze
        neg_aff = torch.squeeze(neg_aff)
        return neg_aff

    def mask_mp_edge_reconstruction(self, feat, mps, train_edge_false_index, train_edge_f_index, train_edge_index, train_edge_ulu_index,train_edge_ullu_index, epoch):
        masked_gs = self.mps_to_gs(mps)
        for i in range(len(masked_gs)):
            #masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_rep, _, emb_mps_list = self.encoder(masked_gs, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)
        feat_recon = rep

        neg_samples_list = []
        for i in range(self.neg_sample_size):
            neg_samples_list.append(feat_recon[train_edge_false_index[:,i]])
        neg_samples = torch.stack(neg_samples_list, dim=1)
        neg_af = self.neg_cost(feat_recon,neg_samples)

        #gs_recon = torch.mm(feat_recon, feat_recon.T)
        input_u_1 = feat_recon[train_edge_f_index]
        af1 = self.affinity(feat_recon, input_u_1)
        del input_u_1
        # input_1 = feat_recon[train_edge_index[:50000,0]]
        input_u2 = feat_recon[train_edge_index]
        af2 = self.affinity(feat_recon, input_u2)
        del input_u2
        input_u3 = feat_recon[train_edge_ulu_index]
        af3 = self.affinity(feat_recon, input_u3)
        del input_u3
        input_u4 = feat_recon[train_edge_ullu_index]
        af4 = self.affinity(feat_recon, input_u4)
        del input_u4
        #loss = None
        true_1_xent = torch.binary_cross_entropy_with_logits(input=af1, target=torch.ones_like(af1))
        true_2_xent = torch.binary_cross_entropy_with_logits(input=af2, target=torch.ones_like(af2))
        true_3_xent = torch.binary_cross_entropy_with_logits(input=af3, target=torch.ones_like(af3))
        true_4_xent = torch.binary_cross_entropy_with_logits(input=af4, target=torch.ones_like(af4))
        neg_xent = torch.binary_cross_entropy_with_logits(input=neg_af, target=torch.zeros_like(neg_af))
        loss = self.true_1_weight*torch.mean(true_1_xent) + self.true_2_weight* torch.mean(true_2_xent)+\
               self.true_3_weight* torch.mean(true_3_xent) +\
               self.true_4_weight*torch.mean(true_4_xent) +self.neg_weight*torch.mean(torch.sum(neg_xent, dim=1))


        return loss


    def get_embeds(self, feats, mps, *varg):
        origin_feat = feats[0]
        gs = self.mps_to_gs(mps)
        rep, _, _ = self.encoder(gs, origin_feat)
        return rep.detach()

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs


def setup_module(m_type, num_metapath, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual,
                 norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "han":
        mod = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod
