from torch.nn import Module, Parameter, CrossEntropyLoss, NLLLoss, Softmax
from torch import nn
import math
import torch


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SoftmaxFace(Module):
    def __init__(
        self,
        head_type=None,
        embedding_size=512,
        classnum=70722,
        s=64.0,
        quality_scale=0,
        t_alpha=0.01,
        h=0.333,
        m=0.4,
    ):
        super(SoftmaxFace, self).__init__()
        self.head_type = head_type
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.CEL = CrossEntropyLoss()
        self.NLLL = NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.s = s
        self.m = m
        # emp prepare
        self.eps = 1e-3
        self.h = h
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.zeros(1))
        self.register_buffer("batch_std", torch.zeros(1))
        self.quality_scale = quality_scale
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def _margin_scaler(self, norms):

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()
        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = (
                mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            )
            self.batch_std = (
                std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
            )

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps
        )  # 66% between -1, 1
        margin_scaler = (
            margin_scaler * self.h
        )  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        return margin_scaler

    def _quality_scaler(self, margin_scaler):
        if self.quality_scale == 0:
            return 1

        # return torch.where(margin_scaler < self.quality_scale, 0, 1)
        # margin_scaler = margin_scaler[:len(margin_scaler) // 2]
        quality_scaler = 1 / (
            1
            + torch.exp(
                -(margin_scaler + 2 * (0.5 - self.quality_scale)) * self.s
            )
        )
        return quality_scaler
        # return torch.cat([quality_scaler, quality_scaler])

    def _cosine(self, embbedings):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability
        return cosine

    def forward(self, embbedings, norms, label):
        # scale
        cosine = self._cosine(embbedings)
        self.cosine = cosine.gather(1, label.view(-1, 1))
        scaled_cosine_m = cosine * self.s
        P_log = self.log_softmax(scaled_cosine_m)
        loss = self.NLLL(P_log, label)
        return loss


class AdaFace(SoftmaxFace):
    def __init__(self, **kwargs):
        super(AdaFace, self).__init__(**kwargs)

    def forward(self, embbedings, norms, label):
        cosine = self._cosine(embbedings)
        self.similarity = cosine.clone().detach().gather(1, label.view(-1, 1))
        margin_scaler = self._margin_scaler(norms)
        # g_angular
        m_arc = torch.zeros(
            label.size()[0], cosine.size()[1], device=cosine.device
        )
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(
            theta + m_arc, min=self.eps, max=math.pi - self.eps
        )
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(
            label.size()[0], cosine.size()[1], device=cosine.device
        )
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        # q, k mirroring in the margin_scaler
        scaled_cosine_m *= self._quality_scaler(margin_scaler)
        P_log = self.log_softmax(scaled_cosine_m)

        loss = self.NLLL(P_log, label)
        return loss, margin_scaler


class CosFace(SoftmaxFace):
    def __init__(self, **kwargs):
        super(CosFace, self).__init__(**kwargs)

    def forward(self, embbedings, norms, label):
        cosine = self._cosine(embbedings)
        margin_scaler = self._margin_scaler(norms)

        m_hot = torch.zeros(
            label.size()[0], cosine.size()[1], device=cosine.device
        )
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        scaled_cosine_m *= self._quality_scaler(margin_scaler)
        loss = self.CEL(scaled_cosine_m, label)
        return loss


class ArcFace(SoftmaxFace):
    def __init__(self, **kwargs):
        super(ArcFace, self).__init__(**kwargs)

    def forward(self, embbedings, norms, label):
        cosine = self._cosine(embbedings)
        margin_scaler = self._margin_scaler(norms)

        m_hot = torch.zeros(
            label.size()[0], cosine.size()[1], device=cosine.device
        )
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)
        theta = cosine.acos()
        theta_m = torch.clip(
            theta + m_hot, min=self.eps, max=math.pi - self.eps
        )
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s
        scaled_cosine_m *= self._quality_scaler(margin_scaler)
        loss = self.CEL(scaled_cosine_m, label)
        return loss



class QGFace(SoftmaxFace):
    def __init__(
        self,
        quality_scale_method="sgn",
        quality_scale_range="half",
        mask_same_class=False,
        detach_pos=False,
        margin_method=None,
        pair_coupling="D2N",
        quality_select_method="low",
        rescale=True,
        **kwargs
    ):
        super(QGFace, self).__init__(**kwargs)
        self.quality_scale_method = quality_scale_method
        self.mask_same_class = mask_same_class
        self.detach_pos = detach_pos
        self.margin_method = margin_method
        self.pair_coupling = pair_coupling
        self.quality_select_method = quality_select_method
        self.quality_scale_range = quality_scale_range
        self.rescale = rescale
        del self.kernel

    def _quality_scaler(self, margin_scaler, norms):

        q_norms, k_norms = norms
        q_margin_scaler, k_margin_scaler = margin_scaler
        q_normal_scaler = (q_margin_scaler + 1) / 2
        k_normal_scaler = (k_margin_scaler + 1) / 2
        if self.quality_select_method == "low":
            target_normal_scaler = torch.min(q_normal_scaler, k_normal_scaler)
            self.norms = torch.min(q_norms, k_norms)
        elif self.quality_select_method == "mean":
            target_normal_scaler = (q_normal_scaler + k_normal_scaler) / 2
            self.norms = (q_norms + k_norms) / 2
        elif self.quality_select_method == "high":
            target_normal_scaler = torch.max(q_normal_scaler, k_normal_scaler)
            self.norms = torch.max(q_norms, k_norms)

        if self.pair_coupling.startswith("D"):
            target_normal_scaler = torch.cat([target_normal_scaler] * 2)
            self.norms = torch.cat([self.norms] * 2)

        if self.quality_scale == 0:
            return 1

        s2 = 0
        if self.quality_scale_method == "sgn":
            s1 = 1
            if self.quality_scale_range == "whole":
                s2 = 1

        elif self.quality_scale_method == "cos":
            s1 = (
                1
                - torch.cos(
                    torch.pi
                    / (self.quality_scale + self.eps)
                    * target_normal_scaler
                )
            ) / 2
            if self.quality_scale_range == "whole":
                s2 = (
                    torch.cos(
                        torch.pi
                        / (1 - self.quality_scale + self.eps)
                        / 2
                        * (
                            target_normal_scaler
                            - self.quality_scale
                            + (1 - self.quality_scale)
                        )
                    )
                    + 1
                )
        elif self.quality_scale_method == "linear":
            s1 = 1 / (self.quality_scale + self.eps) * target_normal_scaler
            if self.quality_scale_range == "whole":
                s2 = (
                    1
                    / (self.quality_scale - 1 + self.eps)
                    * (target_normal_scaler - 1)
                )

        return torch.where(target_normal_scaler < self.quality_scale, s1, s2)

    def supervision_mask(self, labels, cosine, queue_chunk_size):
        q_label, queue_label = labels
        # cosine should be composed like [Nx(K+1)] for [pos, neg]
        idx_matrix = torch.zeros_like(
            cosine, dtype=torch.bool, device=cosine.device
        )
        
        N_qk = q_label.shape[0]
        if not self.mask_same_class:
            queue_chunk_size = min(queue_chunk_size, cosine.shape[1] - 1)
            N_ways = min(queue_chunk_size // N_qk, 1)
            q_label = torch.arange(N_qk, device=q_label.device)
            queue_label = torch.cat([q_label] * N_ways)
            mask = q_label.unsqueeze(1) == queue_label.unsqueeze(0)
        else:
            queue_chunk_size = queue_label.shape[0]
            mask = (q_label.unsqueeze(1) == queue_label.repeat(N_qk, 1))
        if self.pair_coupling.startswith("D"):
            mask = torch.cat([mask, mask])
        idx_matrix[:, 1 : queue_chunk_size + 1] = mask
        cosine[idx_matrix] = 0

    def _m_hot(self, cos_theta, contra_label):
        _m = 0
        # for statistics
        index = torch.arange(contra_label.shape[0], device=cos_theta.device)
        clone_theta = cos_theta.clone().detach()
        pos_target = clone_theta[index, contra_label]
        clone_theta[index, contra_label] = 0
        neg_target = clone_theta.sort(1, True)[0][:, 0]
        one_batch_m = (pos_target - neg_target).mean().item()
        # self.adapt_m = torch.tensor(0, device='cuda:0')
        self.batch_mean = (
            self.t_alpha * one_batch_m + (1 - self.t_alpha) * self.batch_mean
        )
        if self.margin_method == "dynamic":
            _m = self.batch_mean.item()
        elif self.margin_method == "static":
            _m = self.m
        m_hot = torch.zeros_like(cos_theta)
        m_hot.scatter_(1, contra_label.view(-1, 1), _m)
        return m_hot

    ###################################################
    def _cosine(self, embeddings):
        q, k, queue = embeddings
        contra_label = torch.zeros(
            q.shape[0], dtype=torch.long, device=q.device
        )
        q, k = l2_norm(q, axis=1), l2_norm(k, axis=1)
        if self.detach_pos:
            cos_pos = torch.einsum(
                "nc,nc->n", [q, k.clone().detach()]
            ).unsqueeze(-1)
        else:
            cos_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        cos_neg = torch.einsum("nc,kc->nk", [q, queue])
        if self.pair_coupling.startswith("D"):
            contra_label = torch.cat([contra_label, contra_label])
            if self.detach_pos:
                cos_pos = torch.cat(
                    [
                        cos_pos,
                        torch.einsum(
                            "nc,nc->n", [k, q.clone().detach()]
                        ).unsqueeze(-1),
                    ]
                )
            else:
                cos_pos = torch.cat(
                    [cos_pos, torch.einsum("nc,nc->n", [k, q]).unsqueeze(-1)]
                )
            cos_neg = torch.cat(
                [cos_neg, torch.einsum("nc,kc->nk", [k, queue])]
            )

        cosine = torch.cat([cos_pos, cos_neg], dim=1)
        cosine = cosine.clamp(-1, 1)  # for numerical stability
        return cosine, contra_label

    ###################################################

    def forward(
        self, embeddings, norms, labels, margin_scaler, queue_chunk_size
    ):
        # similarity and norms are only calculated with positive pairs
        cosine, contra_label = self._cosine(embeddings)
        self.similarity = (
            cosine.clone().detach().gather(1, contra_label.view(-1, 1))
        )

        self.supervision_mask(labels, cosine, queue_chunk_size)
        m_hot = self._m_hot(cosine, contra_label)
        cosine = cosine - m_hot
        quality_scaler = self._quality_scaler(margin_scaler, norms)
        scaled_cosine_m = cosine * self.s * quality_scaler
        P_log = self.log_softmax(scaled_cosine_m)
        loss = self.NLLL(P_log, contra_label)
        if self.rescale:
            l_down_s = self.CEL(torch.zeros(1, cosine.shape[1]), torch.LongTensor([0]))
            l_up_s = self.CEL(torch.zeros(1, 129), torch.LongTensor([0]))
            loss = loss / l_down_s * l_up_s
        return loss
        # CEL(torch.zeros(1,129), torch.LongTensor([1])) = 4.8598
        # CEL(torch.zeros(1,128), torch.LongTensor([1])) = 4.8520
        # CEL(torch.zeros(1,1024), torch.LongTensor([1])) = 6.932
        # CEL(torch.zeros(1,65537), torch.LongTensor([1])) = 11.0904
        # CEL(torch.zeros(1,8193), torch.LongTensor([1])) = 9.0110
