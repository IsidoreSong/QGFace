import numpy as np
import pandas as pd
import os
import torch

def get_all_files(data_root, aligned_dir):
    lm_df = pd.read_csv(os.path.join(data_root, 'all.txt'), header=None,sep='\t')
    lm_df['img_pth'] = lm_df[0].apply(lambda x: os.path.join(data_root, aligned_dir, x + '.jpg'))
    lm_df = lm_df[~lm_df[0].str.contains('cam8') & ~lm_df[0].str.contains('cam7') & ~lm_df[0].str.contains('cam6')].reset_index(drop=True)
    return lm_df['img_pth'].tolist()


class SCFaceTest:
    def __init__(self, data_root, aligned_dir):
        self.data_root = data_root
        self.lm_df = pd.Series(get_all_files(data_root, aligned_dir))
        self.probe_lm_df = self.lm_df[self.lm_df.str.contains('cam')]
        self.gallery_lm_df = self.lm_df[self.lm_df.str.contains('frontal')]
        self.init_proto()

    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        return int(os.path.basename(image_path).split("_")[0])

    def init_proto(self):
        self.probe_distance_idx_list = [self.probe_lm_df[self.probe_lm_df.str.endswith(f'_{i}.jpg')].index.to_numpy() for i in range(1, 4)]
        self.probe_distance_label_list = [self.probe_lm_df[self.probe_distance_idx_list[i]].apply(lambda x: int(os.path.basename(x).split('_')[0])).to_numpy() for i in range(3)]
        self.indices_gallery = self.gallery_lm_df.index.to_numpy()
        self.labels_gallery = self.gallery_lm_df.apply(lambda x: int(os.path.basename(x).split('_')[0])).to_numpy()

    def test_identification(self, features, ranks=[1], gpu_id=None):
        results_list = []
        feat_gallery = features[self.indices_gallery]
        for i, indices in enumerate(self.probe_distance_idx_list):
            feat_probe = features[indices]
            compare_func = inner_product
            score_mat = compare_func(feat_probe, feat_gallery)
            label_mat = self.probe_distance_label_list[i][:, None] == self.labels_gallery[None, :]
            results, _, __ = DIR_FAR(score_mat, label_mat, ranks, gpu_id=gpu_id)
            results_list.append(results.item())
        return results_list

def inner_product(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    if x1.ndim == 3:
        raise ValueError("why?")
        x1, x2 = x1[:, :, 0], x2[:, :, 0]
    return np.dot(x1, x2.T)


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False, gpu_id=None):
    """
    Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    """
    assert score_mat.shape == label_mat.shape
    # assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[match_indices, :]
    label_mat_m = label_mat[match_indices, :]
    score_mat_nm = score_mat[np.logical_not(match_indices), :]
    label_mat_nm = label_mat[np.logical_not(match_indices), :]

    # print("mate probes: %d, non mate probes: %d" % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
        openset = False
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        openset = True

    # Sort the labels row by row according to scores
    if gpu_id is not None:
        chunk_size = 10000
        score_mat_m_G = torch.tensor(score_mat_m, device=gpu_id)
        sort_idx_mat_m_G = torch.empty_like(score_mat_m_G, dtype=torch.long)
        for i in range(0, score_mat_m.shape[0], chunk_size):
            sort_idx_mat_m_G[i:i + chunk_size, :] = torch.argsort(torch.tensor(score_mat_m[i:i + chunk_size, :], device=gpu_id), dim=1)
        sort_idx_mat_m = sort_idx_mat_m_G.cpu().numpy()
    else:
        sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row, :] = label_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    if openset:
        gt_score_m = score_mat_m[label_mat_m]
        assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    if get_false_indices:
        false_retrieval = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_reject = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_accept = np.zeros([len(FARs), len(ranks), score_mat_nm.shape[0]], dtype=np.bool)
    for i, threshold in enumerate(thresholds):
        for j, rank in enumerate(ranks):
            success_retrieval = sorted_label_mat_m[:, 0:rank].any(axis=1)
            if openset:
                success_threshold = gt_score_m >= threshold
                DIRs[i, j] = (success_threshold & success_retrieval).astype(np.float32).mean()
            else:
                DIRs[i, j] = success_retrieval.astype(np.float32).mean()
            if get_false_indices:
                false_retrieval[i, j] = ~success_retrieval
                false_accept[i, j] = score_mat_nm.max(1) >= threshold
                if openset:
                    false_reject[i, j] = ~success_threshold
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    if get_false_indices:
        return DIRs, FARs, thresholds, match_indices, false_retrieval, false_reject, false_accept, sort_idx_mat_m
    else:
        return DIRs, FARs, thresholds


# Find thresholds given FARs
# but the real FARs using these thresholds could be different
# the exact FARs need to recomputed using calcROC
def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-5):
    #     Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds
