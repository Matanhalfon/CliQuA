import collections
import dataclasses
import re
from typing import Sequence
import numpy as np

from annotated_sample import AnnotatedPassage, AnnotatedSample, ModelAnnotation

@dataclasses.dataclass
class CliQuAStats:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    tp_attributions: int
    max_f1: float


    def __init__(self,
                true_positives: int = 0, 
                false_positives: int = 0,
                true_negatives: int = 0, 
                false_negatives: int = 0,
                tp_attributions: int = 0,
                max_f1 = 0):
        "Initializes a RAGStats object with the provided parameters."
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.tp_attributions = tp_attributions
        self.max_f1 = max_f1


    def calc_precision(self) -> float:
        return self.true_positives / (self.true_positives +
                                       self.false_positives)

    def calc_recall(self) -> float:
        return self.true_positives / (self.true_positives +
                                      self.false_negatives)
    
    def f1(self):
        precision = self.calc_precision()
        recall = self.calc_recall()
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1*100
    

    def total(self):
        return (
            self.true_negatives + self.false_negatives +
            self.true_positives + self.false_positives
            )
    

def normalize4eval(str):
    #new_str = re.sub(r'\\n', ' ', str).strip()
    new_str = re.sub(r'\n+', ' ', str).strip()
    new_str = re.sub(r'\t+', ' ', new_str).strip()
    new_str = re.sub(r' +', ' ', new_str).strip()
    new_str = re.sub(r'"+', '\'', new_str).strip()
    new_str = re.sub(r"“+", '\'', new_str).strip()
    new_str = re.sub(r"&#8220;", '\'', new_str).strip()
    new_str = re.sub(r"&#8221;", '\'', new_str).strip()
    new_str = re.sub(r"”+", '\'', new_str).strip()
    new_str = re.sub(r'`+', '\'', new_str).strip()
    new_str = re.sub(r'\'+', '\'', new_str).strip()
    new_str = re.sub(r'\\n+', ' ', new_str).strip()

    new_str = re.sub(r' +', ' ', new_str).strip()
    new_str = new_str.strip("'")
    new_str = new_str.strip("-")
    return new_str.lower()


def avg_positive_attribution_len(model_annotations: ModelAnnotation) -> float:
    att_lens=[]
    for att in model_annotations:
        if att.annotated_passages[0].model_annotation.label:
            if not att.annotated_passages[0].model_annotation.reference:
                continue
            att_lens.append(len(
                att.annotated_passages[0].model_annotation.reference.split(" ")))

    return np.mean(att_lens)

def calc_stats_record(annotated_samples: Sequence[AnnotatedSample]
             ) -> CliQuAStats:
    "Calcualtes F1 over matches of all samples in sequence."
    stats = CliQuAStats()

    for sample in annotated_samples:
        m_labels = [
            a.model_annotation.label for a in sample.annotated_passages
        ]
        gt_true = sample.ground_truth_label
        m_true = any(m_labels)

        if m_true:
            if gt_true:
                stats.true_positives +=1
            else:
                stats.false_positives += 1
        else: # not m_true
            if gt_true:
                stats.false_negatives += 1
            else:
                stats.true_negatives += 1

    return stats

def valid_attribution_stats(
        annotated_samples: Sequence[AnnotatedSample], 
        require_model_match = True,
        require_model_attribution = True
      ) -> tuple[int, int, float]:
    """
    Calculates and returns valid attribution stats AnnotatedSample sequence.

    Args:
        annotated_samples: A sequence of AnnotatedSample instances.
        require_model_match: If True only passages where model_annotation.label
          is true are included in attribution analysis.
        require_model_attribution: If True only passages where 
        model_annotation.reference is not empty can contribute towards True
        Positive count.

    Returns:
        A tuple containing:
            - int: Total number of true positive samples.
            - int: Total number of true positive samples with valid
                attribution.

    Raises:
        ValueError: If no true positive samples are found.

    This function iterates over the samples, identifying true positives and
    checking if any of their annotated passages have a valid attribution (i.e.,
    the model reference contains any of the ground-truth attribution targets).
    The function returns the count of true positives, count of true positives
    with valid attributions, and the ratio between them.
    """
    
    def _has_valid_attribution(
            annotated_passage: AnnotatedPassage) -> bool:
    
        if not annotated_passage.model_annotation.reference:
            return False
        norm_prediction = normalize4eval(
            annotated_passage.model_annotation.reference).replace(" ","")
        for target in annotated_passage.ground_truth_attributions:
            if not target:
                continue
            norm_target = normalize4eval(target).replace(" ","")
            if  norm_target in norm_prediction:
                return True
        return False

    def _f1_score(prediction: str, target:str) -> float:
        prediction_tokens = normalize4eval(prediction).split()
        target_tokens = normalize4eval(target).split()
        common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(target_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _max_f1(annotated_passage: AnnotatedPassage) -> float:
        max_f1 = 0
        for target in annotated_passage.ground_truth_attributions:
            if not target:
                target = ""
            prediction = annotated_passage.model_annotation.reference
            if not prediction:
                prediction = ""
            f1 = _f1_score(prediction, target)
            if f1 > max_f1:
                max_f1 = f1
        return max_f1

    tp_count = 0
    tp_with_valid_attribution = 0
    f1_scores = []
    for sample in annotated_samples:
        valid_attribution_found = False

        m_labels = [
            a.model_annotation.label for a in sample.annotated_passages
            ]
        if require_model_match and not any(m_labels):
            continue
        qualified_tp = False
        max_f1_sample = 0
        for passage in sample.annotated_passages:
            # # Skip passages without ground truth attributions.
            # if len(passage.ground_truth_attributions) < 1:
            #     continue
            model_annotations = passage.model_annotation
            if require_model_match and not model_annotations.label:
                continue
            if require_model_attribution and not model_annotations.reference:
                continue
            if sample.ground_truth_label:
                qualified_tp = True
                if _has_valid_attribution(passage):
                    valid_attribution_found = True
            passage_f1 = _max_f1(passage)
            if max_f1_sample < passage_f1:
                max_f1_sample = passage_f1
        f1_scores.append(max_f1_sample)
        tp_count += 1 if qualified_tp else 0
        tp_with_valid_attribution += 1 if valid_attribution_found else 0
    return (tp_count, tp_with_valid_attribution, np.mean(f1_scores))


def calc_stats(annotated_samples: Sequence[AnnotatedSample],
               valid_attributions: bool = True,
               require_model_match:bool = True,
               require_model_attributions:bool = True) -> CliQuAStats:

    stats = calc_stats_record(annotated_samples)
    if valid_attributions:
        (_, stats.tp_attributions, stats.max_f1) = valid_attribution_stats(
            annotated_samples, require_model_match, require_model_attributions)
    return stats



def attribution_grounding_stats(
        samples: Sequence[AnnotatedSample],
        split_indicators: Sequence[str] = []) -> tuple[int, int, int, int]:
    positive_ungrouneded = 0
    positive_empty = 0
    negative_ungrounded = 0
    negative_empty = 0
    for sample in samples:
        for passage in sample.annotated_passages:
            model_annotation = passage.model_annotation
            empty = False if model_annotation.reference else True
            if empty:
               positive_empty += 1 if model_annotation.label else 0
               negative_empty += 0 if model_annotation.label else 1
               continue
            references = [model_annotation.reference]
            for split_indicator in split_indicators:
                references = references[0].split(split_indicator)
                if len(references) > 1:
                    break
            for refernce in references:
                grounded = (normalize4eval(refernce).replace(" ", "") in 
                            normalize4eval(passage.passage).replace(" ", ""))
                if not grounded: break
            if grounded:
                continue
            positive_ungrouneded += 1 if model_annotation.label else 0
            negative_ungrounded += 0 if model_annotation.label else 1
    return (positive_ungrouneded, positive_empty, negative_ungrounded,
             negative_empty)
