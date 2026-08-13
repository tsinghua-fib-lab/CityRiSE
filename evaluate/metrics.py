import json
import csv
import argparse
import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

def compute_regression_metrics(pred_list, true_list):
    mse = mean_squared_error(true_list, pred_list)
    mae = mean_absolute_error(true_list, pred_list)
    r2 = r2_score(true_list, pred_list)
    rmse = mean_squared_error(true_list, pred_list, squared=False)
    mean_true = sum(true_list) / len(true_list)
    nrmse = rmse / mean_true if mean_true != 0 else None  
    return mse, mae, r2, rmse, nrmse


def extract_float(value):
    if isinstance(value, (float, int)):
        return float(value)

    if isinstance(value, str):
        if len(value) > 100:
            return None

        value_str = value.replace(",", "").replace("$", "").replace("£", "").replace(" ", "")

        try:
            return float(value_str)
        except ValueError:
            return None

    return None

def extract_float_res(response):
    if not isinstance(response, str):
        print(f"[ERROR] response is not a string! type: {type(response)}, value: {response}")
        return None
    match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE)
    if match:
        answer_str = match.group(1).strip()
        try:
            return float(answer_str)
        except ValueError:
            return None
    else:
        return None


def calculate_r2_sft(response_file):
    with open(response_file, 'r') as f:
        data = json.load(f)

    true_vals = []
    pred_vals = []

    for item in data:
        
        true_val = extract_float(item['rank_group'])
        response = item['response']
        
        pred_val = extract_float(response)

        if true_val is None or pred_val is None or \
        str(true_val).lower() == "nan" or str(pred_val).lower() == "nan":
            print(f"Warning: Invalid data found in item: {item['response']}")
            continue
        if pred_val < 0:
            continue

        true_vals.append(true_val)
        pred_vals.append(pred_val)
    print("length of vals: ", len(true_vals), len(pred_vals))
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print("No valid data found.")
    else:
        mse, mae, r2, rmse, nrmse = compute_regression_metrics(pred_vals, true_vals)
        print("r2:", r2)
    return mse, mae, r2, rmse, nrmse



def calculate_r2(response_file):
    with open(response_file, 'r') as f:
        data = json.load(f)

    true_vals = []
    pred_vals = []

    for item in data:
        
        true_val = extract_float(item['rank_group'])
        response = item['response']
        
        pred_val = extract_float_res(response)
        
        if true_val is None or pred_val is None or \
        str(true_val).lower() == "nan" or str(pred_val).lower() == "nan":
            continue
        if pred_val < 0:
            continue

        true_vals.append(true_val)
        pred_vals.append(pred_val)
    print("length of vals: ", len(true_vals), len(pred_vals))
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print("No valid data found.")
    else:
        mse, mae, r2, rmse, nrmse = compute_regression_metrics(pred_vals, true_vals)
        print("r2:", r2)
    return mse, mae, r2, rmse, nrmse


def calculate_acc_sft(response_file):
    with open(response_file, 'r') as f:
        data = json.load(f)
    total = 0
    correct = 0
    format_error = 0
    counts = defaultdict(int)
    response_counts = defaultdict(int)
    correct_samples = []
    wrong_samples = []
    city_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        city = item['city']
        response = item.get('response', '')

        reference = item.get('rank_group')
        counts[reference] += 1
        if not isinstance(response, str):
            pred = -2
            print(f"response not str: {response}")
            format_error += 1
        else:
            
            pred = response
            response_counts[pred] += 1
            
        is_correct = (float(pred) == float(reference))
        total += 1
        correct += int(is_correct)

        city_stats[city]["total"] += 1
        if is_correct:
            city_stats[city]["correct"] += 1

        else:
            print("reference:", reference, "pred:", pred)


    acc = correct / total if total > 0 else 0
    print(f"accuracy: {acc:.4f} ({correct}/{total})")
    print(f"format error: {format_error}")
    for city, stats in sorted(city_stats.items()):
        correct = stats["correct"]
        total = stats["total"]
        acc_city = correct / total if total > 0 else 0

    return acc


def calculate_acc(response_file):
    with open(response_file, 'r') as f:
        data = json.load(f)
    total = 0
    correct = 0
    format_error = 0
    counts = defaultdict(int)
    response_counts = defaultdict(int)
    correct_samples = []
    wrong_samples = []
    city_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data:
        city = item['city']
        response = item.get('response', '')

        reference = item.get('rank_group')
        counts[reference] += 1
        if not isinstance(response, str):
            pred = -2
            print(f"response: {response}")
            format_error += 1
        else:
            match = re.search(r'(?<![:\w])<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)

            if match:
                pred = match.group(1).upper().strip() if match else None
                response_counts[pred] += 1
            else:
                pred = -2
                format_error += 1
        is_correct = (float(pred) == float(reference))
        total += 1
        correct += int(is_correct)

        city_stats[city]["total"] += 1
        if is_correct:
            city_stats[city]["correct"] += 1

        else:
            print("reference:", reference, "pred:", pred)


    acc = correct / total if total > 0 else 0
    print(f"accuracy: {acc:.4f} ({correct}/{total})")
    print(f"format error: {format_error}")
    
    for city, stats in sorted(city_stats.items()):
        correct = stats["correct"]
        total = stats["total"]
        acc_city = correct / total if total > 0 else 0
        
    return acc


def record_metrics(acc, summary_path, model_name, eval_version, mse, mae, r2, rmse, nrmse):
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'accuracy', 'eval_version', 'mse', 'mae', 'r2', 'rmse', 'nrmse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'model_name': model_name, 'accuracy': f"{acc:.4f}", 'eval_version': eval_version, 'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse, 'nrmse': nrmse})
    print(f"have been recorded to {summary_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2_5_vl_7b_comparison_cot_addr_relation-v2.1-epoch1-sft', help='model name')
    parser.add_argument('--task_name', type=str, default='gdp', help='task name')
    parser.add_argument('--training_mode', type=str, default='rl', help='sft or rl')
    parser.add_argument('--domain', type=str, default='out', help='in or out')
    parser.add_argument('--num_process', type=int, default=20, help='number of processes for evaluation')
    parser.add_argument('--eval_version', type=str, default=20, help='evaluate data version')

    args = parser.parse_args()
    model_name_full = args.model_name.replace("/", "_")
    response_path = f'results/training/{args.training_mode}/{args.domain}/{model_name_full}_{args.task_name}_response-{args.eval_version}-10.json'
    if "gemini" in args.model_name:
        response_path = f'results/training/{args.training_mode}/{args.domain}/{model_name_full}_{args.task_name}_response-{args.eval_version}-10.json'
    summary_path = f'results/training/{args.training_mode}/{args.domain}/{args.task_name}_response.csv'

    if args.training_mode == "rl":
        mse, mae, r2, rmse, nrmse = calculate_r2(response_path)
    elif args.training_mode == "sft":
        mse, mae, r2, rmse, nrmse = calculate_r2_sft(response_path)
    
    record_metrics(acc, summary_path, args.model_name, args.eval_version, mse, mae, r2, rmse, nrmse)

