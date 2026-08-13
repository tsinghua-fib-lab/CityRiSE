# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import numpy as np

from mathruler.grader import grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)


    content_match = re.search(r"<answer>(.*?)</answer>", response)
    given_answer = content_match.group(1).strip() if content_match else response.strip()
    if "mental" in response or "public" in response or "bachelor" in response or "population" in response or "gdp" in response:
        
        try:
            given_answer = float(given_answer)
            
            score = 0.5 if format_match else 0.0

            bonus_tags = ["person", "vehicle", "greenery", "road infrastructure", "street furniture", "building"]
            for tag in bonus_tags:
                if tag in response:
                    score += 0.05
            if "location" in response:
                score += 0.2 

            return min(score, 1.0)
        except ValueError:
            score = 1.0 if format_match else 0.0
            return score
    else:
        score = 1.0 if format_match else 0.0
        return score


def exp_huber_reward(y_true: float, y_pred: float, delta=3.0, alpha=0.8) -> float:

    error = y_pred - y_true
    abs_error = abs(error)

    if abs_error <= delta:
        huber_loss = 0.5 * error ** 2
    else:
        huber_loss = delta * (abs_error - 0.5 * delta)

    reward = np.exp(-alpha * huber_loss)
    return reward

def accuracy_reward(response: str, ground_truth: str) -> float:
    content_match = re.search(r"<answer>(.*?)</answer>", response)
    given_answer = content_match.group(1).strip() if content_match else response.strip()
    try:
        
        pred_val = float(given_answer)
        true_val = float(ground_truth)
        # print(f"true_val: {true_val}, pred_val: {pred_val}")
        acc_reward = exp_huber_reward(true_val, pred_val)
        print(f"acc_reward: {acc_reward}")

        return acc_reward

    except ValueError:
        if grade_answer(given_answer, ground_truth):
            return 1.0


    return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")

    response = reward_input["response"]

    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
