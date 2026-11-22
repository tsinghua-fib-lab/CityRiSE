import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluate import metrics


@pytest.mark.parametrize("true,pred", [
    ([1, 2, 3], [1, 2, 3]),
    ([1, 2, 4], [1, 0, 5]),
    ([0.1, 0.25, 1.25], [0.2, 0.35, 1.35]),
    ([-3, -1, -2], [-2, 1, -1]),
    ([-1, 0, 1], [-1, 0, 2]),
])
def test_regression_matches_numpy_reference(true, pred):
    observed = metrics.compute_regression_metrics(pred, true)
    expected_true = np.array(true)
    errors = np.array(pred) - expected_true
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mean_true = np.mean(expected_true)
    r2 = 1 - np.sum(errors**2) / np.sum((expected_true - mean_true)**2)
    nrmse = rmse / mean_true if mean_true else None
    assert observed[:4] == pytest.approx((mse, np.mean(np.abs(errors)), r2, rmse))
    if nrmse is None:
        assert observed[4] is None
    else:
        assert observed[4] == pytest.approx(nrmse)


def write_responses(path, truths, responses):
    rows = [
        {"city": ("city_a", "city_b")[i % 2], "rank_group": truth, "response": response}
        for i, (truth, response) in enumerate(zip(truths, responses))
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


@pytest.mark.parametrize("score,responses,correct", [
    (metrics.calculate_acc, ["<answer> 1 </answer>", "<answer>1e0</answer>", "<answer>3.0</answer>"], 2),
    (metrics.calculate_acc_sft, [" 1 ", "1e0", "3.0"], 2),
    (metrics.calculate_acc, ["<answer>1</answer>", "<answer>1</answer>", "no tag",
                             "<answer> </answer>", "<answer>unparseable</answer>"], 1),
    (metrics.calculate_acc_sft, ["1", "1", "", "unparseable", "--"], 1),
])
def test_accuracy_denominator_and_response_formats(tmp_path, score, responses, correct):
    path = tmp_path / "responses.json"
    write_responses(path, range(1, len(responses) + 1), responses)
    assert score(path) == pytest.approx(correct / len(responses))


@pytest.mark.parametrize("score,responses", [
    (metrics.calculate_acc, ["<answer> $1,000 </answer>", "<answer>£ 1,001.00</answer>"]),
    (metrics.calculate_acc_sft, ["  $1,000  ", "£ 1,001.00"]),
])
def test_accuracy_uses_existing_numeric_normalization(tmp_path, score, responses):
    path = tmp_path / "responses.json"
    write_responses(path, [1000, "1,001"], responses)
    assert score(path) == 1.0


CSV_FIELDS = ["model_name", "accuracy", "eval_version", "mse", "mae", "r2", "rmse", "nrmse"]


@pytest.mark.parametrize("mode", ["rl", "sft"])
def test_documented_cli_writes_and_appends_summary(tmp_path, mode):
    result_dir = tmp_path / "results" / "training" / mode / "out"
    result_dir.mkdir(parents=True)
    summary = result_dir / "gdp_response.csv"
    checkout = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (checkout, env.get("PYTHONPATH"))))

    def run(version, predictions):
        path = result_dir / f"fixture_gdp_response-{version}-10.json"
        responses = [f"<answer>{pred}</answer>" if mode == "rl" else pred for pred in predictions]
        write_responses(path, [1, 2, 3], responses)
        response_bytes = path.read_bytes()
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-m", "evaluate.metrics", "--model_name=fixture",
             "--task_name=gdp", "--domain=out", f"--eval_version={version}", f"--training_mode={mode}"],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert path.read_bytes() == response_bytes

    target_rmse = np.sqrt(0.5)
    run(20, ["1", "1", "unparseable"])
    with summary.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == CSV_FIELDS
        initial_rows = list(reader)
    assert len(initial_rows) == 1
    assert initial_rows[0]["accuracy"] == "0.3333"
    assert [float(initial_rows[0][field]) for field in CSV_FIELDS[3:]] == pytest.approx(
        [0.5, 0.5, -1.0, target_rmse, target_rmse / 1.5]
    )

    run(21, ["1", "2", "unparseable"])
    with summary.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0] == initial_rows[0]
    assert [(row["model_name"], row["eval_version"]) for row in rows] == [("fixture", "20"), ("fixture", "21")]
    assert rows[1]["accuracy"] == "0.6667"
    assert [float(rows[1][field]) for field in CSV_FIELDS[3:]] == [0.0, 0.0, 1.0, 0.0, 0.0]
