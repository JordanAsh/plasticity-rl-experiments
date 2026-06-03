import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string (last <answer>...</answer>)."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def validate_equation(equation_str, available_numbers):
    """Equation must use each available number exactly once."""
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r"\d+", equation_str))
        return numbers_in_eq == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate an arithmetic-only expression."""
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """Countdown task scoring.

    Args:
        solution_str: model response
        ground_truth: dict {"target": int, "numbers": list[int]}
        format_score: partial credit for well-formed but wrong answer
        score: credit for correct answer

    Returns:
        0 if no <answer>, format_score if invalid/wrong, score if correct.
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")

    if equation is None:
        return 0

    if not validate_equation(equation, numbers):
        return format_score

    result = evaluate_equation(equation)
    if result is None:
        return format_score

    if abs(result - target) < 1e-5:
        return score
    return format_score
