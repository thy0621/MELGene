from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import os
import torch
import torch.nn.functional as F

class KBCModel(torch.nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < chunk_size:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)
                    scores = q @ rhs
                    targets = self.score(these_queries)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        # print(queries[b_begin + i, 2].item() == query[2].item())  True
                        filter_out += [queries[b_begin + i, 2].item()]

                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6  # filter
                    ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()
                    b_begin += batch_size  # next batch

                c_begin += chunk_size
        return ranks

    def scores(self, q, gene_num):
        rhs = self.get_rhs(0, gene_num)
        lhs = self.lhs(q[:, 0])
        rel = self.rel(q[:, 1])
        print(rhs.shape, lhs.shape, rel.shape)
        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        output
        scores = output @ self.rhs.weight.t()
        return scores

    def get_predicted_gene(self, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]], gene_num: int,
                           device: str = 'cpu'):
        with torch.no_grad():
            disease = torch.tensor([]).to(device)
            gene = torch.tensor([]).to(device)
            score = torch.tensor([]).to(device)
            rhs = self.get_rhs(0, gene_num)
            q = self.get_queries(queries)
            scores = q @ rhs
            for i, query in enumerate(queries):
                if (query[0].item(), query[1].item()) in filters:
                    in_train_gene = filters[(query[0].item(), query[1].item())]
                    all_gene_score = scores[i]
                    all_gene_score[torch.LongTensor(in_train_gene)] = -1e6

                    gene_rank = torch.argsort(all_gene_score, descending=True)[:(gene_num - len(in_train_gene))]
                else:
                    all_gene_score = scores[i]
                    gene_rank = torch.argsort(scores[i], descending=True)[:gene_num]
                gene_rank_scores = all_gene_score[gene_rank].view(-1, 1)

                disease = torch.cat((disease, torch.full(size=(len(gene_rank), 1), fill_value=query[0].item(),
                                                         dtype=torch.long).to(device)), 0)
                gene = torch.cat((gene, gene_rank.view(-1, 1)), 0)
                score = torch.cat((score, gene_rank_scores.view(-1, 1)), 0)
        return disease, gene, score


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = torch.nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def save(self, fold: int):
        os.makedirs('../models/cp', exist_ok=True)
        torch.save(self, f'../models/cp/fold_{fold}.pt')


class KDGene(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], edim: int, rdim: int, gatecell: str,
            init_size: float = 1e-3,
    ):
        super(KDGene, self).__init__()
        self.sizes = sizes
        self.edim = edim
        self.rdim = rdim
        self.gatecell = gatecell

        self.lhs = torch.nn.Embedding(sizes[0], edim, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rdim, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], edim, sparse=True)

        self.gate = {
            'RNNCell': lambda: torch.nn.RNNCell(rdim, edim),
            'LSTMCell': lambda: torch.nn.LSTMCell(rdim, edim),
            'GRUCell': lambda: torch.nn.GRUCell(rdim, edim)
        }[gatecell]()

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)

        return torch.sum(lhs * rel_update * rhs, 1, keepdim=True)


    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        output = lhs * rel_update
        pred = output @ self.rhs.weight.t()
        return pred, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        return lhs * rel_update

    def save(self, fold: int):
        os.makedirs('../models/kdg', exist_ok=True)
        torch.save(self, f'../models/kdg/fold_{fold}.pt')


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size


    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def save(self, fold: int):
        os.makedirs('../models/complex', exist_ok=True)
        torch.save(self, f'../models/complex/fold_{fold}.pt')

class DistMult(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(DistMult, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(s, rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        return (lhs * rel) @ self.embeddings[0].weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        return lhs * rel

    def save(self, fold: int):
        os.makedirs('../models/distmult', exist_ok=True)
        torch.save(self, f'../models/distmult/fold_{fold}.pt')


class N3(torch.nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight  # 0.01

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]

class RESCAL(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.dim = rank
        #self.norm = norm  # 使用L1范数还是L2范数
        #self.alpha = alpha

        # 实体和关系嵌入
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sizes[0], rank, sparse=True),  # 实体嵌入
            torch.nn.Embedding(sizes[1], rank * rank, sparse=True)  # 关系嵌入
        ])

        # 初始化实体向量
        torch.nn.init.xavier_uniform_(self.embeddings[0].weight.data)
        self.embeddings[0].weight.data = F.normalize(self.embeddings[0].weight.data, 2, 1)

        # 初始化关系矩阵
        torch.nn.init.xavier_uniform_(self.embeddings[1].weight.data)
        self.embeddings[1].weight.data = F.normalize(self.embeddings[1].weight.data, 2, 1)

    def forward(self, x):
        h_embs = self.embeddings[0](x[:, 0])
        r_mats = self.embeddings[1](x[:, 1])
        t_embs = self.embeddings[0](x[:, 2])

        r_mats = r_mats.view(-1, self.dim, self.dim)
        t_embs = t_embs.view(-1, self.dim, 1)

        scores = (h_embs.unsqueeze(1) @ r_mats).squeeze(1) @ self.embeddings[0].weight.t()

        return scores, (h_embs, r_mats, t_embs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]).view(-1, self.dim, self.dim)
        return lhs * rel

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]).view(-1, self.dim, self.dim)
        rhs = self.embeddings[0](x[:, 2]).view(-1, self.dim, 1)
        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def save(self, fold: int):
        os.makedirs('../models/rescal', exist_ok=True)
        torch.save(self, f'../models/rescal/fold_{fold}.pt')


class HoLE(KBCModel):

    def __init__(self, **kwargs):
        super(HoLE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "cmax", "cmin"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        self.ent_embeddings = NamedEmbedding("ent_embedding", self.tot_entity, self.hidden_size)
        self.rel_embeddings = NamedEmbedding("rel_embedding", self.tot_relation, self.hidden_size)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.pairwise_hinge

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        r_e = F.normalize(r_e, p=2, dim=-1)
        h_e = torch.stack((h_e, torch.zeros_like(h_e)), -1)
        t_e = torch.stack((t_e, torch.zeros_like(t_e)), -1)
        e, _ = torch.unbind(torch.ifft2(torch.conj(torch.fft.fft2(h_e, 1)) * torch.fft.fft2(t_e, 1), 1), -1)
        return -F.sigmoid(torch.sum(r_e * e, 1))

    def embed(self, h, r, t):
        """
            Function to get the embedding value.
            Args:
                h (Tensor): Head entities ids.
                r  (Tensor): Relation ids of the triple.
                t (Tensor): Tail entity ids of the triple.
            Returns:
                tuple: Returns a 3-tuple of head, relation and tail embedding tensors.
        """
        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.ent_embeddings(t)
        return emb_h, emb_r, emb_t

class SimplE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(SimplE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.emb_sizes = (sizes[1], sizes[1], sizes[0], sizes[2])
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(s, rank, sparse=True)
            for s in self.emb_sizes
        ])
        for i in range(4):
            self.embeddings[i].weight.data *= init_size

    def score(self, x):
        rel = self.embeddings[0](x[:, 1])
        rel_inv = self.embeddings[1](x[:, 1])
        lhs = self.embeddings[2](x[:, 0])
        rhs = self.embeddings[3](x[:, 2])

        forward = torch.sum(lhs * rel * rhs, 1, keepdim=True)
        backward = torch.sum(lhs * rel_inv * rhs, 1, keepdim=True)

        return 0.5 * (forward+backward)

    def forward(self, x):
        rel = self.embeddings[0](x[:, 1])
        rel_inv = self.embeddings[1](x[:, 1])
        lhs = self.embeddings[2](x[:, 0])
        rhs = self.embeddings[3](x[:, 2])

        f = (lhs * rel) @ self.embeddings[3].weight.t()
        b = (rhs * rel_inv) @ self.embeddings[2].weight.t()
        scores = 0.5 * (f+b)

        return scores, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[3].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[2](queries[:, 0])
        rel = self.embeddings[0](queries[:, 1])
        return lhs * rel

    def save(self, fold: int):
        os.makedirs('../models/simple', exist_ok=True)
        torch.save(self, f'../models/simple/fold_{fold}.pt')