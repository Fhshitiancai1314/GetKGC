import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig
from helper import get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.prompter import Prompter
from models.bert_for_layerwise import BertModelForLayerwise
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class KGCPromptTuner(pl.LightningModule):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)
        if self.configs.graph_model in ['transe', 'rotate']:
            self.rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)

        self.prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.tail_prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)
        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)
        if configs.n_lar > 0:
            self.lar_loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=configs.gamma,
                distance_function=lambda x, y: self.graph_model.score_fn(x, y[0], y[1]),
            )

        self.history = {'perf': ..., 'loss': []}
        self.loss_fn = get_loss_fn(configs)
#         self.supconloss = SupConLoss(temperature=configs.temp1, contrast_mode="all", base_temperature=configs.temp1).to(torch.device("cuda"))
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        if self.configs.alpha_step > 0:
            self.alpha = 0.
        else:
            self.alpha = self.configs.alpha

    def forward(self, ent_rel, src_ids, src_mask, labels, tail_src_ids, tail_src_mask):
        all_ent_embed = self.ent_embed.weight
        if self.configs.graph_model in ['transe', 'rotate']:
            all_rel_embed = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            all_rel_embed = self.rel_embed.weight

        ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        ent_embed = all_ent_embed[ent]
        print(ent_embed.size())
        rel_embed = all_rel_embed[rel]
        prompt = self.prompter(torch.stack([ent_embed, rel_embed], dim=1))
        prompt_attention_mask = torch.ones(ent_embed.size(0), self.configs.prompt_length * 2).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)

        # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * 2]
        plm_ent_embed, plm_rel_embed = torch.chunk(ent_rel_state, chunks=2, dim=1)
        plm_ent_embed = self.fc(plm_ent_embed.reshape(ent_embed.size(0), -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(rel_embed.size(0), -1))

        tail_embed = all_ent_embed[labels]
        tail_prompt = self.tail_prompter(tail_embed)
        tail_prompt_attention_mask = torch.ones(tail_embed.size(0), self.configs.prompt_length).type_as(src_mask)
        tail_src_mask = torch.cat((tail_prompt_attention_mask, tail_src_mask), dim=1)
        output = self.plm(input_ids=tail_src_ids, attention_mask=tail_src_mask, layerwise_prompt=tail_prompt)

        last_hidden_state = output.last_hidden_state
        plm_tail_embed = last_hidden_state[:, :self.configs.prompt_length]
        plm_tail_embed = self.fc(plm_tail_embed.reshape(tail_embed.size(0), -1))

        # pred -- .shape: (batch_size, embed_dim)
        pred = self.graph_model(plm_ent_embed, plm_rel_embed)
        # logits -- .shape: (batch_size, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)
        return logits, pred, plm_tail_embed

    def training_step(self, batched_data, batch_idx):
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        # print("batched_data:", batched_data)
        # print("source_ids:", batched_data['source_ids'], batched_data['source_ids'].shape)
        # print("source_mask:", batched_data['source_mask'], batched_data['source_mask'].shape)
        # print("ent_rel:", batched_data['ent_rel'], batched_data['ent_rel'].shape)
        # print("tgt_ent:", batched_data['tgt_ent'], len(batched_data['tgt_ent']))
        # print("labels:", batched_data['labels'], len(batched_data['labels']))
        # print("lars:", batched_data['lars'], batched_data['lars'].shape)

        tail_src_ids = batched_data['tail_source_ids']
        tail_src_mask = batched_data['tail_source_mask']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        tgt_ent = batched_data['tgt_ent']
        labels = batched_data['labels']
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        lars = batched_data['lars'] if self.configs.n_lar > 0 else None

        logits, pred, tail_embed = self(ent_rel, src_ids, src_mask, labels, tail_src_ids, tail_src_mask)

        x1_node = F.normalize(pred, dim=1)
        tail_embed = F.normalize(tail_embed, dim=1)

        # calculate SupCon loss
        features1 = torch.cat((x1_node.unsqueeze(1), tail_embed.unsqueeze(1)), dim=1)
        # features2 = torch.cat((x2_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)
        # SupCon Loss
#         supconloss1 = self.supconloss(features1, labels=labels, mask=None)
        celoss = self.loss_fn(logits, labels)
        loss = celoss
        if self.configs.n_lar > 0:
            lar_ent_embed = self.ent_embed
            # pos, neg -- .shape: (batch_size, 1, embed_dim), pos_bias, neg_bias -- .shape: (batch_size, 1)
            pos, lar = lar_ent_embed(labels).unsqueeze(1), torch.mean(lar_ent_embed(lars), dim=1, keepdim=True)
            pos_bias, lar_bias = self.graph_model.bias[labels].unsqueeze(-1), torch.mean(self.graph_model.bias[lars], dim=-1, keepdim=True)
            lar_loss = self.lar_loss_fn(anchor=pred, positive=(pos, pos_bias), negative=(lar, lar_bias))
            loss = loss + self.alpha * lar_loss

        self.history['loss'].append(loss.detach().item())
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # test_triples .shape: (batch_size, 3)
        test_triples = batched_data['triple']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # tgt_ent -- .type: list
        tgt_ent = batched_data['tgt_ent']
        
        tail_src_ids = batched_data['tail_source_ids']
        tail_src_mask = batched_data['tail_source_mask']
        
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        logits, _, _ = self(ent_rel, src_ids, src_mask, tgt_ent, tail_src_ids, tail_src_mask)
        logits = logits.detach()
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt[(hi, ri)]
            ## store target score
            tgt_score = logits[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            ## recover the target values
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, argsort, ranks, batch_idx)
        return ranks

    def validation_epoch_end(self, outs):
        tail_ranks = np.concatenate(outs[0])
        head_ranks = np.concatenate(outs[1])

        perf = get_performance(self, tail_ranks, head_ranks)
        print('Epoch:', self.current_epoch)
        print(perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.configs.lr)

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx):
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                triple_str = self.ent_names[head] + ' [' + self.ent_descs[head] + '] | ' + self.rel_names[rel]\
                    + ' | ' + self.ent_names[tail] + ' [' + self.ent_descs[tail] + '] ' + timestamp + '(%d %d %d)' % (head, tail, rel)
                file.write(str(batch_idx * self.configs.val_batch_size + i) + '. ' + triple_str + '=> ranks: ' + str(rank) + '\n')

                best10 = argsort[i, :10]
                for ii, ent in enumerate(best10):
                    ent = ent.item()
                    mark = '*' if (ii + 1) == rank else ' '
                    file.write('\t%2d%s ' % (ii + 1, mark) + self.ent_names[ent] + ' [' + self.ent_descs[ent] + ']' + ' (%d)' % ent + '\n')
