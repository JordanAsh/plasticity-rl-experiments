"""Knights and Knaves logic puzzle reward function.

Faithful to Logic-RL (https://github.com/Unakar/Logic-RL):
- format_score = +1 if exactly one each of <think>, </think>, <answer>, </answer>
                  appear in that order; else -1
- answer_score = +2 if predicted person->role dict matches ground truth
                 -1.5 if dict can be parsed but is wrong
                 -2 if dict cannot be parsed OR format was wrong

Range: total in [-3, +3]. Format-gaming alone yields +1-2 = -1, so it's
penalized. Real solving yields +1+2 = +3.
"""
import re
from typing import Dict, Optional, Tuple


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract content of the last <answer> tag from the model response."""
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed_str = solution_str

    matches = list(re.finditer(r"<answer>(.*?)</answer>", processed_str, re.DOTALL))
    if not matches:
        return None, processed_str
    return matches[-1].group(1).strip(), processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    status_dict = {}
    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    knight_count = answer_text.lower().count("knight")
    knave_count = answer_text.lower().count("knave")
    if knight_count + knave_count != len(expected_names):
        return None
    status_dict = {}
    for name in expected_names:
        pattern = re.compile(
            rf"\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b", re.IGNORECASE
        )
        match = pattern.search(answer_text)
        if not match:
            return None
        status_dict[name] = match.group(1).lower()
    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    positions = []
    for tag in tags:
        if processed_str.count(tag) != 1:
            return False
        positions.append(processed_str.find(tag))
    return positions == sorted(positions)


def compute_score(solution_str: str, ground_truth: Dict,
                  format_reward: float = 1.0, answer_reward: float = 2.0) -> float:
    """Logic-RL scoring. Range: [-3, +3]."""
    solution_text = ground_truth.get("solution_text_format", "")
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())

    answer_text, processed_str = extract_solution(solution_str)
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    if format_correct and answer_text is not None:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status is None:
            answer_score = -abs(answer_reward)
        elif pred_status == gt_status:
            answer_score = answer_reward
        else:
            answer_score = -1.5
    else:
        answer_score = -abs(answer_reward)

    return format_score + answer_score

