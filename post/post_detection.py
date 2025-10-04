import os
import torch
import numpy as np
from utils import DETECTOR_LIST, DETECTOR_RES_DIR, FINAL_RES_DIR

score_key_list = [
    'cl_score', 'belt_score',
    'ours_hardware_-1_cl_0', 'ours_hardware_-1_cl_1', 'ours_hardware_-1_cl_2',
    'ours_hardware_0_cl_0', 'ours_hardware_0_cl_1', 'ours_hardware_0_cl_2'
]


METRIC_KEYS = [
    'TN_rate', 'FP_rate', "FN_rate", "TP_rate", 'F1_score'
]

DET_FIN_RES = os.path.join(FINAL_RES_DIR, 'det')
os.makedirs(DET_FIN_RES, exist_ok=True)


def load_det_res(task_id, detect_id):
    det_cls_name = DETECTOR_LIST[detect_id].__name__

    task_dir = os.path.join(DETECTOR_RES_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    save_path = os.path.join(task_dir, f"{det_cls_name}.det")
    res = torch.load(save_path)
    return res



def post_detect_1():
    detect_id = 0
    final_res = []
    det_cls_name = DETECTOR_LIST[detect_id].__name__
    save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}.csv")
    for task_id in range(6):
        res = load_det_res(task_id=task_id, detect_id=detect_id)
        res = np.array([res[k] for k in score_key_list]).reshape([-1, 1])
        final_res.append(res)
    final_res = np.concatenate(final_res, axis=1)
    final_res = np.where(final_res == None, '', final_res)  # For string-based placeholders

    np.savetxt(save_path, final_res, delimiter=',', fmt='%s')


def post_detect_2():

    def process_score_func(field):
        if field is None:
            return np.zeros([10000, 2])
        return torch.cat(
            (field['benign_scores'].reshape([-1, 1]), field['bd_scores'].reshape([-1, 1])),
            axis=1).numpy()

    def process_p_func(field):
        if field is None:
            return np.array([None, None]).reshape([1, -1])
        return np.array([field['statistic'], field['p_value']]).reshape([1, -1])

    detect_id = 1
    final_scores = []
    final_p = []
    det_cls_name = DETECTOR_LIST[detect_id].__name__
    score_save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}_score.csv")
    p_save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}_p.csv")
    for task_id in range(6):
        res = load_det_res(task_id=task_id, detect_id=detect_id)

        score = np.concatenate([process_score_func(res[k]) for k in score_key_list], axis=1)
        p_v = np.concatenate([process_p_func(res[k]) for k in score_key_list])

        final_scores.append(score)
        final_p.append(p_v)
    final_scores = np.concatenate(final_scores, axis=1)
    final_p = np.concatenate(final_p, axis=1)
    final_scores = np.where(final_scores == None, '', final_scores)  # For string-based placeholders
    final_p = np.where(final_p == None, '', final_p)
    np.savetxt(score_save_path, final_scores, delimiter=',', fmt='%s')
    np.savetxt(p_save_path, final_p, delimiter=',', fmt='%s')

def post_detect_3():
    def process_score_func(field):
        if field is None:
            return None
        return max(field['score'])

    def process_metric_func(field):
        if field is None:
            return np.array([None for k in METRIC_KEYS]).reshape([1, -1])
        return np.array([field[k] for k in METRIC_KEYS]).reshape([1, -1])

    detect_id = 2
    final_score = []
    final_metric = []
    det_cls_name = DETECTOR_LIST[detect_id].__name__
    score_save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}_score.csv")
    metric_save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}_metric.csv")
    for task_id in range(6):
        res = load_det_res(task_id=task_id, detect_id=detect_id)
        score = np.array([process_score_func(res[k]) for k in score_key_list]).reshape([-1, 1])
        metric = np.concatenate([process_metric_func(res[k]) for k in score_key_list])
        final_score.append(score)
        final_metric.append(metric)

    final_score = np.concatenate(final_score, axis=1)
    final_score = np.where(final_score == None, '', final_score)  # For string-based placeholders
    np.savetxt(score_save_path, final_score, delimiter=',', fmt='%s')

    final_metric = np.concatenate(final_metric, axis=1)
    final_metric = np.where(final_metric == None, '', final_metric)  # For string-based placeholders
    np.savetxt(metric_save_path, final_metric, delimiter=',', fmt='%s')


def post_detect_4():
    def process_score_func(field):
        if field is None:
            return None
        return field['score']


    detect_id = 3
    final_score = []

    det_cls_name = DETECTOR_LIST[detect_id].__name__
    score_save_path = os.path.join(DET_FIN_RES, f"{det_cls_name}_score.csv")
    for task_id in range(6):
        res = load_det_res(task_id=task_id, detect_id=detect_id)
        score = np.array([process_score_func(res[k]) for k in score_key_list]).reshape([-1, 1])

        final_score.append(score)


    final_score = np.concatenate(final_score, axis=1)
    final_score = np.where(final_score == None, '', final_score)  # For string-based placeholders
    np.savetxt(score_save_path, final_score, delimiter=',', fmt='%s')

    # final_metric = np.concatenate(final_metric, axis=1)
    # final_metric = np.where(final_metric == None, '', final_metric)  # For string-based placeholders
    # np.savetxt(metric_save_path, final_metric, delimiter=',', fmt='%s')



if __name__ == '__main__':
    # post_detect_1()
    post_detect_2()