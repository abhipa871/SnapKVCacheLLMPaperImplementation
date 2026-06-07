"""
LongBench metrics and scoring (vendored from SnapKV experiments/LongBench).

Source: https://github.com/FasterDecoding/SnapKV/tree/main/experiments/LongBench
"""

from __future__ import annotations

import re
import string
from collections import Counter

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge

dataset2metric = {
    "narrativeqa": "qa_f1_score",
    "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score",
    "multifieldqa_zh": "qa_f1_zh_score",
    "hotpotqa": "qa_f1_score",
    "2wikimqa": "qa_f1_score",
    "musique": "qa_f1_score",
    "dureader": "rouge_zh_score",
    "gov_report": "rouge_score",
    "qmsum": "rouge_score",
    "multi_news": "rouge_score",
    "vcsum": "rouge_zh_score",
    "trec": "classification_score",
    "triviaqa": "qa_f1_score",
    "samsum": "rouge_score",
    "lsht": "classification_score",
    "passage_retrieval_en": "retrieval_score",
    "passage_count": "count_score",
    "passage_retrieval_zh": "retrieval_zh_score",
    "lcc": "code_sim_score",
    "repobench-p": "code_sim_score",
}

LINE_POSTPROCESS_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》"
            "「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction_line = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction_line = line
            break
    return fuzz.ratio(prediction_line, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


_METRIC_FUNCS = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}


def _postprocess_prediction(prediction: str, dataset: str) -> str:
    if dataset in LINE_POSTPROCESS_DATASETS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def scorer(dataset: str, predictions, answers, all_classes) -> float:
    """Return dataset score on 0-100 scale (SnapKV eval.py)."""
    metric_name = dataset2metric[dataset]
    metric_fn = _METRIC_FUNCS[metric_name]
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        prediction = _postprocess_prediction(prediction, dataset)
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, metric_fn(prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)
