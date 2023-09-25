import json

# This file is used to reproduce the results of the DSTC9 Track 1 evaluation metrics.

ref_data = json.load(open("data/dstc9/test/labels.json", 'rt'))
hyp_data = json.load(open("results/results_dstc9/baseline/entry0.json", 'rt'))
human_data = json.load(open("results/results_dstc9/baseline/entry0.human_eval.json", 'rt'))

_detection_tp = 0.0
_detection_fp = 0.0
_detection_fn = 0.0

def _compute(score_sum):
    if _detection_tp + _detection_fp > 0.0:
        score_p = score_sum/(_detection_tp + _detection_fp)
    else:
        score_p = 0.0

    if _detection_tp + _detection_fn > 0.0:
        score_r = score_sum/(_detection_tp + _detection_fn)
    else:
        score_r = 0.0

    if score_p + score_r > 0.0:
        score_f = 2*score_p*score_r/(score_p+score_r)
    else:
        score_f = 0.0

    return (score_p, score_r, score_f)


score = 0.0
for human, hyp, ref in zip(human_data, hyp_data, ref_data):
    if ref['target'] is True:
        if hyp['target'] is True:
            _detection_tp += 1.0
            score += sum(human['accuracy']) / 3
        else:
            _detection_fn += 1.0
    else:
        if hyp['target'] is True:
            _detection_fp += 1.0

print(_detection_tp)

print(_compute(score))