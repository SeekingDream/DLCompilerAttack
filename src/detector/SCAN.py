'''
Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection

This file is modified based on the following source:
link : https://github.com/TDteach/Demon-in-the-Variant/blob/master/pysrc/SCAn.py
The detection method is called SCAn.
@inproceedings{tang2021demon,
    title={Demon in the variant: Statistical analysis of $\{$DNNs$\}$ for robust backdoor contamination detection},
    author={Tang, Di and Wang, XiaoFeng and Tang, Haixu and Zhang, Kehuan},
    booktitle={30th USENIX Security Symposium (USENIX Security 21)},
    pages={1541--1558},
    year={2021}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. SCAn detection:
        a. Leverage the target model to generate representations for all input images.
        b. Estimate the parameters by running an EM algorithm.
        c. calculate the identity vector and decompose the representations.
        d. estimate the parameters for the mixture model.
        e. perform the likelihood ratio test.
    4. compute TPR and FPR

'''

import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score


from .abst_detector import AbstDetector


def compute_metrics(predict, ground_truth):
    """
    Computes TN, FP, FN, TP rates and the F1-score from binary arrays.

    Args:
        predict (np.ndarray): Predicted binary array.
        ground_truth (np.ndarray): Ground truth binary array.

    Returns:
        dict: A dictionary containing TN, FP, FN, TP rates and the F1-score.
    """
    if len(predict) != len(ground_truth):
        raise ValueError("Predict and ground_truth arrays must have the same length.")

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predict).ravel()

    # Compute rates
    total = tn + fp + fn + tp
    tn_rate = tn / total
    fp_rate = fp / total
    fn_rate = fn / total
    tp_rate = tp / total

    # Compute F1-score
    f1 = f1_score(ground_truth, predict)

    return {
        "TN_rate": tn_rate,
        "FP_rate": fp_rate,
        "FN_rate": fn_rate,
        "TP_rate": tp_rate,
        "F1_score": f1
    }

EPS = 1e-5
class SCAn:
    def __init__(self):
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k, i_sc[0][0], np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        SeG = np.matmul(Se,G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(SeG, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N


class SCANDetector(AbstDetector):

    def __init__(self, clean_loader, test_loader, device):
        super().__init__(clean_loader, test_loader, device)
        self.num_per_label = 10

    @torch.no_grad()
    def get_features_labels(self, model, target_layer, data_loader, bd_trigger):
        def feature_hook(module, input_, output_):
            global feature_vector
            feature_vector = input_
            return None

        h = target_layer.register_forward_hook(feature_hook)

        model.eval()
        features = []
        labels = []

        for batch_idx, batch in tqdm(enumerate(data_loader)):
            inputs, targets = batch['input'].to(self.device), batch['label'].to(self.device)

            global feature_vector
            if bd_trigger is not None:
                inputs = bd_trigger.add_trigger(inputs)
            outputs = model(inputs)
            # feature_vector = torch.sum(torch.flatten(feature_vector, 2), 2)
            if isinstance(feature_vector, tuple):
                feature_vector = feature_vector[0]
            current_feature = feature_vector.detach().cpu().numpy()
            current_labels = targets.cpu().numpy()

            # Store features
            features.append(current_feature)
            labels.append(current_labels)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        h.remove()  # Rmove the hook

        return features, labels


    def find_last_linear_layer(self, module):
        """
        Recursively traverses the model to find the last non-linear layer before the classification layer.
        """
        children = list(module.children())
        if not children:  # If there are no more submodules, return the module itself
            return module
        if isinstance(children[-1], nn.Linear):  # If the last layer is Linear, return the penultimate layer
            return children[-1] if len(children) > 1 else module
        # Recurse into the last child
        return self.find_last_linear_layer(children[-1])



    def compute_score(self, model, bd_trigger):
        NC = model.class_num
        model = model.to(self.device).eval()

        target_layer = self.find_last_linear_layer(model)
        if isinstance(target_layer, nn.Sequential):
            target_layer = target_layer[-1]
        clean_features, clean_labels = self.get_features_labels(
            model, target_layer, self.clean_loader, None)
        train_cl_features, train_cl_labels = self.get_features_labels(
            model, target_layer, self.test_loader, None)

        train_bd_features, train_bd_labels = self.get_features_labels(
            model, target_layer, self.test_loader, bd_trigger)

        feats_inspection = np.concatenate([train_cl_features, train_bd_features])
        class_indices_inspection = np.concatenate([train_cl_labels, train_bd_labels])

        feats_clean = np.array(clean_features)
        class_indices_clean = np.array(clean_labels)

        ground_truth_label = np.zeros(len(feats_inspection))
        ground_truth_label[-len(train_bd_features):] = 1

        scan = SCAn()
        gb_model = scan.build_global_model(feats_clean, class_indices_clean, NC)
        size_inspection_set = len(feats_inspection)
        feats_all = np.concatenate([feats_inspection, feats_clean])
        class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])
        lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, NC)
        score = scan.calc_final_score(lc_model)
        threshold = np.exp(2)

        suspicious_indices = []
        flag_list = []

        for target_class in range(NC):

            print('[class-%d] outlier_score = %f' % (target_class, score[target_class]))

            if score[target_class] <= threshold:
                continue
            flag_list.append([target_class, score[target_class]])
            tar_label = (class_indices_all == target_class)
            all_label = np.arange(len(class_indices_all))
            tar = all_label[tar_label]

            cluster_0_indices = []
            cluster_1_indices = []

            cluster_0_clean = []
            cluster_1_clean = []

            for index, i in enumerate(lc_model['subg'][target_class]):
                if i == 1:
                    if tar[index] > size_inspection_set:
                        cluster_1_clean.append(tar[index])
                    else:
                        cluster_1_indices.append(tar[index])
                else:
                    if tar[index] > size_inspection_set:
                        cluster_0_clean.append(tar[index])
                    else:
                        cluster_0_indices.append(tar[index])

            if len(cluster_0_clean) < len(cluster_1_clean):  # if most clean samples are in cluster 1
                suspicious_indices += cluster_0_indices
            else:
                suspicious_indices += cluster_1_indices
        detect_labels = np.zeros_like(ground_truth_label)
        for i in suspicious_indices:
            detect_labels[i] = 1
        res = compute_metrics(detect_labels, ground_truth_label)
        res['score'] = np.log(score)
        return res


