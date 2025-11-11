import torch
from torch import nn, autograd
from SimKGC_vectors import Reachability
import json
import numpy as np
from torchsummary import summary

class Selector(nn.Module):
    def __init__(self, cfg, nbf, simkgc, rotate, complex, kdg, distmult, simple, num_layers, hidden_dims, input_dim, train=False):
        super(Selector, self).__init__()

        self.nbf = nbf
        if (not train):
            self.nbf.requires_grad_(False)
        if (simkgc is not None):
            self.simkgc = simkgc
        if (rotate is not None):
            self.rotate = rotate
            self.rotate.requires_grad_(False)
            self.tot_rels = self.rotate.remb.size(0)
        if (complex is not None):
            self.complex = complex
            self.complex.requires_grad_(False)
        if (kdg is not None):
            self.kdg = kdg
            self.kdg.requires_grad_(False)
        if (distmult is not None):
            self.distmult = distmult
            self.distmult.requires_grad_(False)
        if (simple is not None):
            self.simple = simple
            self.simple.requires_grad_(False)

        self.cfg = cfg
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            mlp.append(layer)
            mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        mlp.append(layer)
        mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

        rotate_mlp = []
        rotate_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            rotate_mlp.append(layer)
            rotate_mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        rotate_mlp.append(layer)
        rotate_mlp.append(nn.ReLU())
        self.rotate_mlp = nn.Sequential(*rotate_mlp)

        simkgc_mlp = []
        simkgc_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            simkgc_mlp.append(layer)
            simkgc_mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        simkgc_mlp.append(layer)
        simkgc_mlp.append(nn.ReLU())
        self.simkgc_mlp = nn.Sequential(*simkgc_mlp)

        kdg_mlp = []
        kdg_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            kdg_mlp.append(layer)
            kdg_mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        kdg_mlp.append(layer)
        kdg_mlp.append(nn.ReLU())
        self.kdg_mlp = nn.Sequential(*kdg_mlp)

        distmult_mlp = []
        distmult_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            distmult_mlp.append(layer)
            distmult_mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        distmult_mlp.append(layer)
        distmult_mlp.append(nn.ReLU())
        self.distmult_mlp = nn.Sequential(*distmult_mlp)

        simple_mlp = []
        simple_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            simple_mlp.append(layer)
            simple_mlp.append(nn.ReLU())
        layer = nn.Linear(hidden_dims[-1], 1)
        if (cfg.model.init != 0.0):
            torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
        simple_mlp.append(layer)
        simple_mlp.append(nn.ReLU())
        self.simple_mlp = nn.Sequential(*simple_mlp)


    def get_rotate_scores(self, all_h, r_head):
        all_h = all_h[:, 0]
        r_head = r_head[:, 0]
        h_emb = self.rotate.eemb[all_h].squeeze()
        rule_emb = self.rotate.remb[r_head]
        rule_emb = self.rotate.project(rule_emb)
        h_emb = self.rotate.product(h_emb, rule_emb)
        t_emb = self.rotate.eemb
        h_emb = torch.unsqueeze(h_emb, 1)
        dists = self.rotate.diff(h_emb, t_emb).sum(dim=-1)
        return -dists
    
    def get_rotate_rep(self, all_h, r_head):
        all_h = all_h[:, 0]
        r_head = r_head[:, 0]
        rotate = self.rotate
        h_emb = rotate.eemb[all_h].squeeze()
        rule_emb = rotate.remb[r_head]
        rule_emb = rotate.project(rule_emb)
        h_emb = rotate.product(h_emb, rule_emb)
        return h_emb

    def weights_normalize(self, weights):
        max_abs, _ = torch.max(torch.abs(weights), dim=1, keepdim=True)
        norm = weights/max_abs
        return norm

    def get_features_and_normalize(self, score, t_index):
        pre_maxs = torch.amax(score, dim=1).unsqueeze(dim = -1)
        pre_mins = torch.amin(score, dim=1).unsqueeze(dim = -1)
        pre_diffs = pre_maxs - pre_mins
        score = (score - pre_mins)
        maxs = torch.amax(score, dim=1).unsqueeze(dim = -1)
        score = score/maxs
        var = torch.var(score, dim=1).unsqueeze(dim = -1)
        means = torch.mean(score, dim=1).unsqueeze(dim = -1)
        std = torch.std(score, dim=1).unsqueeze(dim=-1)
        mdiffs = 1 - means
        topk = torch.mean(torch.topk(score, dim=1, k=10, largest=True).values, dim=1).unsqueeze(dim = -1)
        topk_std = torch.std(torch.topk(score, dim=1, k=10, largest=True).values, dim=1).unsqueeze(dim = -1)
        prop = (torch.sum((score >= 0.8), axis = -1)/score.size(1)).unsqueeze(dim = -1)
        # index = torch.topk(score, k=10, dim=-1, largest=True).indices
        # add = torch.zeros(score.shape, device = batch.device)
        # add[:, index] = 0.1
        # score += add
        if self.training:
            score = torch.gather(score, -1, t_index)
        return score, [mdiffs, var]
        # return score, [mdiffs, std, 1 - prop]

    def forward(self, data, batch, edge_weight=None, hard_wts=None):
        h_index, t_index, r_index = batch.unbind(-1)
        '''
        if self.training:
            data, mask = self.nbf.remove_easy_edges(data, h_index, t_index, r_index)
            #训练时去除简单边，原理类似dropout
        '''
        shape = h_index.shape
        #shape没有使用
        h_index, t_index, r_index = self.nbf.negative_sample_to_tail(h_index, t_index, r_index)
        #使用nbfnet的函数将三元组转换为尾实体预测
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        #确保hr一致，对不同尾实体打分
        features = []
        #输入到模型的特征（均值方差连接）
        if (self.cfg.model.need_nbf):
            nbf_output = self.nbf.bellmanford(data, h_index[:, 0], r_index[:, 0])
            feature = nbf_output["node_feature"]
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # feature = feature.gather(1, index)

            # probability logit for each tail node in the batch
            # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
            nbf_score = self.nbf.mlp(feature).squeeze(-1)
            #获取分数
            nbf_score, feature = self.get_features_and_normalize(nbf_score, t_index)
            #这一步进行了归一化，计算分数的均值、方差
            features += feature
            #将均值方差加到features末尾
        
        if (self.cfg.model.need_sim):
            simkgc_score = self.simkgc.get_h(h_index, r_index).to(h_index.device)
            simkgc_score, feature = self.get_features_and_normalize(simkgc_score, t_index)
            features += feature

        if (self.cfg.model.need_rotate):
            rotate_score = self.get_rotate_scores(h_index, r_index)
            rotate_score, feature = self.get_features_and_normalize(rotate_score, t_index)
            features += feature

        if (self.cfg.model.need_complex ):
            complex_score = self.complex.get_score(h_index[:, 0], r_index[:, 0])
            complex_score, feature = self.get_features_and_normalize(complex_score, t_index)
            features += feature

        if (self.cfg.model.need_kdg):
            kdg_score = self.kdg.get_score(h_index[:, 0], r_index[:, 0])
            kdg_score, feature = self.get_features_and_normalize(kdg_score, t_index)
            features += feature

        if (self.cfg.model.need_distmult):
            distmult_score = self.distmult.get_score(h_index[:, 0], r_index[:, 0])
            distmult_score, feature = self.get_features_and_normalize(distmult_score, t_index)
            features += feature
        if (self.cfg.model.need_simple):
            simple_score = self.simple.get_score(h_index[:, 0], r_index[:, 0])
            simple_score, feature = self.get_features_and_normalize(simple_score, t_index)
            features += feature
        if (self.cfg.model.method == 'rerank'):
            if (self.cfg.model.rerank == 'forward'):
                vals = torch.topk(nbf_score, 100, sorted=True).values
                vals = vals[:, -1].unsqueeze(-1)
                mask = nbf_score >= vals

                return nbf_score + mask*1000.0*(1  + simkgc_score), None
            else:
                vals = torch.topk(simkgc_score, 100, sorted=True).values
                vals = vals[:, -1].unsqueeze(-1)
                mask = simkgc_score >= vals

                return simkgc_score + mask*1000.0*(1  + nbf_score), None

        if (self.cfg.model.method == 'ensemble'):
            mlp_in = torch.cat(tuple(features), dim = 1)
            weights = []
            keys = []

            if (self.cfg.model.get_feat):
                return mlp_in, None

            if (self.cfg.model.ensemble_nbf):
                weight = self.mlp(mlp_in.to(h_index.device))
                weights.append(weight)
                keys.append('nbf')
                # print('nbf weight', weight[:4, :])

            if (self.cfg.model.ensemble_sim):
                '''
                outputs = []
                def hook_fn(module, input, output):
                    outputs.append(output)
                for layer in self.simkgc_mlp:
                    layer.register_forward_hook(hook_fn)'''

                simkgc_weight = self.simkgc_mlp(mlp_in.to(h_index.device))
                weights.append(simkgc_weight)
                keys.append('simkgc')
                '''
                for i, output in enumerate(outputs):
                    print(f"Layer {i} output shape: {output.shape}")
                    print(f"Layer {i} output: {output}")'''
                # print('simkgc weight', simkgc_weight[:4, :])

            if (self.cfg.model.ensemble_atomic):
                rotate_weight = self.rotate_mlp(mlp_in.to(h_index.device))
                weights.append(rotate_weight)
                keys.append('rotate')
                # print(rotate_weight[:4, :])



            if (self.cfg.model.ensemble_kdg):
                kdg_weight = self.kdg_mlp(mlp_in.to(h_index.device))
                weights.append(kdg_weight)
                keys.append('kdg')

            if (self.cfg.model.ensemble_distmult):
                distmult_weight = self.distmult_mlp(mlp_in.to(h_index.device))
                weights.append(distmult_weight)
                keys.append('distmult')

            if (self.cfg.model.ensemble_simple):
                simple_weight = self.simple_mlp(mlp_in.to(h_index.device))
                weights.append(simple_weight)
                keys.append('simple')

            weights = torch.cat(weights, dim=-1)
            #weights = self.weights_normalize(weights)
            ws = torch.unbind(weights, dim=-1)
            weights = {}
            i = 0
            for key in keys:
                weights[key] = ws[i].unsqueeze(-1)
                i += 1
            result = 0
            if (self.cfg.model.need_nbf):
                if (self.cfg.model.ensemble_nbf):
                    result += weights['nbf']*nbf_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[0]
                    result += wt*nbf_score
            if (self.cfg.model.need_sim):
                if (self.cfg.model.ensemble_sim):
                    result += weights['simkgc']*simkgc_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[1]
                    result += wt*simkgc_score
            if (self.cfg.model.need_rotate):
                if (self.cfg.model.ensemble_atomic):
                    result += weights['rotate']*rotate_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[2]
                    # wt = torch.FloatTensor(simkgc_score.shape).uniform_(0.5, 1.0).to(r_index.device)
                    result += wt*rotate_score
            if (self.cfg.model.need_complex):
                if (self.cfg.model.ensemble_atomic):
                    result += weights['rotate']*complex_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[3]
                    result += wt*complex_score
            if (self.cfg.model.need_kdg):
                if (self.cfg.model.ensemble_kdg):
                    result += weights['kdg']*kdg_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[4]
                    result += wt*kdg_score
            if (self.cfg.model.need_distmult):
                if (self.cfg.model.ensemble_distmult):
                    result += weights['distmult']*distmult_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[5]
                    result += wt * distmult_score
            if (self.cfg.model.need_simple):
                if (self.cfg.model.ensemble_simple):
                    result += weights['simple']*simple_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[5]
                    result += wt * simple_score
            if (self.cfg.weight_path is not None):
                return result, None
                return result, weights
            else:
                return result, None

    def get_predicted_gene(self, dis_rel, gene_in_train, gene_num=8947, device='cpu'):
        """
        预测疾病与基因之间的关系。

        参数:
            dis_rel (torch.Tensor): 疾病-关系对，形状为 (num_diseases, 2)。
            gene_in_train (torch.Tensor): 训练集中的基因（尾实体）。
            gene_num (int): 基因的总数。
            device (str): 运行设备（如 'cpu' 或 'cuda'）。

        返回:
            disease (torch.Tensor): 疾病 ID。
            gene (torch.Tensor): 预测的基因 ID。
            score (torch.Tensor): 预测得分。
        """
        # 将输入数据移动到指定设备
        dis_rel = dis_rel.to(device)

        # 提取疾病和关系
        h_index = dis_rel[:, 0]  # 疾病 ID
        r_index = dis_rel[:, 1]  # 关系 ID

        # 生成所有可能的尾实体（基因）
        t_index = torch.arange(gene_num, device=device)  # 所有基因 ID

        # 构建批次数据
        h_index = h_index.unsqueeze(-1).repeat(1, gene_num)
        t_index = t_index.unsqueeze(-1).repeat(1, len(h_index)).t()
        r_index = r_index.unsqueeze(-1).repeat(1, gene_num)
        batch = torch.stack((h_index, t_index, r_index), dim=-1)  # 形状为 (num_diseases * gene_num, 3)

        # 调用 forward 方法计算得分
        with torch.no_grad():  # 禁用梯度计算
            scores, _ = self.forward(data=None, batch=batch)  # 使用 forward 方法计算得分
            scores = scores.view(len(dis_rel), -1)  # 重塑为 (num_diseases, gene_num)

        # 过滤掉训练集中已知的基因
        mask = torch.ones_like(scores, dtype=torch.bool)
        for i, disease_id in enumerate(dis_rel[:, 0]):
            if disease_id.item() in gene_in_train:
                known_genes = gene_in_train[disease_id.item()]  # 获取当前疾病在训练集中已知的基因
                mask[i, known_genes] = False  # 将已知基因的得分掩码为 False
        scores[~mask] = -float('inf')  # 将已知基因的得分设置为负无穷

        # 获取预测结果
        top_scores, top_indices = torch.topk(scores, k=100, dim=-1)  # 取前 100 个得分最高的基因
        disease = dis_rel[:, 0].unsqueeze(-1).expand(-1, 100).reshape(-1)  # 疾病 ID
        gene = top_indices.reshape(-1)  # 基因 ID
        score = top_scores.reshape(-1)  # 得分

        return disease, gene, score

