import re

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None:
        return False

    pred = given_answer.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]    
    
    return pred == ground_truth

# def grade_answer(given_answer: str, ground_truth: str) -> bool:
#     if given_answer is None:
#         return False

#     pred = given_answer.replace(",", "")
#     regex = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
#     match = [s for s in regex.findall(pred)]
#     if match:
#         match = match[-1]
#         if isinstance(match, tuple):
#             match = [m for m in match if m][0]
#         match = match.strip()
#     return match == ground_truth
