import os
import subprocess
import tempfile
from pathlib import Path

import yaml


def make_test_config(tmp_path: Path) -> Path:
    # base YAML
    repo_root = Path(__file__).parent
    base_cfg = repo_root / "configs" / "final_resnet_model_config.yml"
    base = yaml.safe_load(base_cfg.read_text())

    # shorten training so it finishes quickly
    tc = base.copy()
    tc["stardist"]["model_config"]["epochs"] = 1
    tc["stardist"]["model_config"]["steps_per_epoch"] = 1
    tc["stardist"]["model_config"]["train_n_val_patches"] = 1

    # point at your mini data
    data_dir = repo_root / "data"
    all_files = sorted([p for p in data_dir.iterdir() if p.is_file()])
    # leave one out for validation/prediction
    training = [str(p) for p in all_files][:-1]
    validation = [training[-1]]
    prediction = [training[-1]]

    tc["data"]["training"] = training
    tc["data"]["validation"] = validation
    tc["data"]["prediction"] = prediction

    # create a fresh temp dir for output
    # (we use tmp_path as parent so that pytest will clean it up)
    out_dir = Path(tempfile.mkdtemp(prefix="stardist_out_", dir=str(tmp_path)))
    tc["data"]["output_dir"] = str(out_dir)
    tc["stardist"]["model_dir"] = str(out_dir.parent)
    tc["stardist"]["model_name"] = str(out_dir.name)

    # write out a temp config
    out_cfg = tmp_path / "test_config.yml"
    out_cfg.write_text(yaml.safe_dump(tc))
    print(yaml.safe_dump(tc))
    return out_cfg


def run_entry(entry: str, cfg_path: Path, env: dict):
    proc = subprocess.run(
        [entry, "--config", str(cfg_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"\n--- {entry} STDOUT ---\n{proc.stdout}")
    print(f"\n--- {entry} STDERR ---\n{proc.stderr}")
    assert proc.returncode == 0, f"{entry} failed (exit code {proc.returncode})"


def test_train_then_predict(tmp_path):
    cfg = make_test_config(tmp_path)

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["CUDA_VISIBLE_DEVICES"] = ""

    # 1) Train the model (1 epoch, 1 step)
    run_entry("train-stardist", cfg, env)

    # 2) Predict using the model just trained
    run_entry("predict-stardist", cfg, env)
