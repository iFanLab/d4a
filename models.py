import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from layers import SLMPConv
from utils import central_moment_discrepancy


class D4A(nn.Module):
    def __init__(self, in_feats, out_feats, n_hid=64, n_layers=2,
                 dropout=0.5, device='cpu', data=None,
                 input_norm=None, norm=None,
                 **kwargs):
        super(D4A, self).__init__()
        self.name = 'd4a'

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.n_hid = n_hid
        self.dropout = dropout
        self.dropout_fn = nn.Dropout(p=dropout)
        self.act_fn = nn.ReLU()
        self.device = device

        ''' Smooth-less Message Passing '''
        self.aggregator_type = kwargs.get('aggregator_type', 'max')
        self.alpha = kwargs.get('alpha', 1.0)
        self.slmp_kw = {
            'aggregator_type': self.aggregator_type,
            'alpha': self.alpha,
            'with_bias': kwargs.get('with_bias', False),
        }

        ''' Distribution Shift Constraint '''
        self.lam_dsc = kwargs.get('lam_dsc', 1)
        self.K_cmd = kwargs.get('K_cmd', 5)

        assert self.aggregator_type in ['max', 'mean', 'sum']

        self.with_input_norm = True
        if input_norm == 'ln':
            self.input_norm = nn.LayerNorm(in_feats)
        elif input_norm == 'bn':
            self.input_norm = nn.BatchNorm1d(in_feats)
        else:
            self.with_input_norm = False

        self.with_norm = True
        if norm == 'ln':
            norm_cls = nn.LayerNorm
        elif norm == 'bn':
            norm_cls = nn.BatchNorm1d
        else:
            self.with_norm = False

        self.gcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if n_layers == 1:
            self.gcs.append(SLMPConv(in_feats, out_feats, **self.slmp_kw))
        else:
            self.gcs.append(SLMPConv(in_feats, n_hid, **self.slmp_kw))
            if self.with_norm:
                self.norms.append(norm_cls(n_hid))
            for i in range(1, n_layers - 1):
                self.gcs.append(SLMPConv(n_hid, n_hid, **self.slmp_kw))
                if self.with_norm:
                    self.norms.append(norm_cls(n_hid))
            self.gcs.append(SLMPConv(n_hid, out_feats, **self.slmp_kw))

        self.reset_parameters()

    def reset_parameters(self):
        if self.with_input_norm:
            self.input_norm.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for gc in self.gcs:
            gc.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.adj
        if self.with_input_norm:
            x = self.input_norm(x)
        for i, gc in enumerate(self.gcs[:-1]):
            x = gc(x, edge_index)
            if self.with_norm:
                x = self.norms[i](x)
            else:
                x = F.normalize(x, p=2, dim=1)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.gcs[-1](x, edge_index)
        return outs

    def fit(self, data, train_mask, val_mask, test_mask, args, verbose=False, **kwargs):
        self.reset_parameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = 0
        best_weight = None
        bad_cnt = 0
        for epoch in range(args.n_epochs):
            self.train()
            outs = self.forward(data)
            optimizer.zero_grad()
            loss = self.loss_fn(outs, data.y, train_mask)
            acc = (outs[train_mask].argmax(1) == data.y[train_mask]).sum().item() / train_mask.sum().item()
            loss.backward()
            optimizer.step()

            # Validation
            eval_dict = self.evaluate(data, val_mask)
            loss_val, acc_val = eval_dict['loss'], eval_dict['acc']

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch+1:04d} loss_train= {loss:.5f} acc_train= {acc:.5f}",
                      f"loss_val= {loss_val:.5f} acc_val= {acc_val:.5f}")

            if acc_val >= best_acc:
                bad_cnt = 0
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_weight = self.state_dict()
            else:
                bad_cnt += 1
                if bad_cnt >= args.patience:
                    break

        self.load_state_dict(best_weight)

    def get_data(self, data: Data):
        edge_index, _ = remove_self_loops(data.edge_index)
        return Data(
            x=data.x.to(self.device),
            adj=edge_index.to(self.device),
            y=data.y.to(self.device)
        )

    def loss_fn(self, outs, y, mask):
        loss = F.cross_entropy(outs[mask], y[mask])

        if (self.lam_dsc != 0) and (self.training):
            preds = F.softmax(outs, dim=-1)
            loss += self.lam_dsc * self.CMD_loss(preds, mask)
        return loss

    def CMD_loss(self, x, train_mask):
        x1 = x[train_mask]
        x2 = x[~train_mask]
        loss_distr_shift = central_moment_discrepancy(x1, x2, self.K_cmd, p=2)
        return sum(loss_distr_shift)

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data)

    @torch.no_grad()
    def evaluate(self, data, mask):
        eval_dict = dict()
        logits = self.predict(data)
        y = data.y
        eval_dict['loss'] = self.loss_fn(logits, y, mask).item()
        eval_dict['acc'] = (logits[mask].argmax(1) == y[mask]).sum().item() / y[mask].size(0)
        return eval_dict

    def set_attrs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, f'get_{k}'):
                setattr(self, k, getattr(self, f'get_{k}')(v))
            else:
                if isinstance(v, torch.Tensor):
                    setattr(self, k, v.to(self.device))
                else:
                    setattr(self, k, v)

    def __repr__(self):
        return f"{self.name}[SLMP=({self.aggregator_type},{self.alpha}), DSC=({self.lam_dsc},{self.K_cmd})]"
