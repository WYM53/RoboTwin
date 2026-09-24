### task_config/dp_test_demo.yml
conda activate robotwin_dp
cd ~/RoboTwin

### 采集数据
bash collect_data.sh place_empty_cup dp_test_demo 0

### 检查 observation
python - <<'PY'
import h5py

p = "/home/dyl216/RoboTwin/data/place_empty_cup/dp_test_demo/data/episode0.hdf5"

with h5py.File(p, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:60s} shape={obj.shape} dtype={obj.dtype}")

    f.visititems(show)
PY

### 把 50 个 HDF5 episode 合并成 DP 使用的 Zarr 数据集
cd ~/RoboTwin/policy/DP
bash process_data.sh place_empty_cup dp_test_demo 50

### 检查 Zarr
python - <<'PY'
import zarr

p = "data/place_empty_cup-dp_test_demo-50.zarr"
root = zarr.open(p, mode="r")

print(root.tree())

print("\n===== shapes =====")
print("head_camera :", root["data/head_camera"].shape)
print("state       :", root["data/state"].shape)
print("action      :", root["data/action"].shape)
print("episode_ends:", root["meta/episode_ends"].shape)

print("\n===== checks =====")
print("state dim :", root["data/state"].shape[1])
print("action dim:", root["data/action"].shape[1])
print("episodes  :", len(root["meta/episode_ends"]))
print("total steps:", root["meta/episode_ends"][-1])
PY

### 训练
conda activate robotwin_dp
cd ~/RoboTwin/policy/DP

bash train.sh place_empty_cup dp_test_demo 50 0 14 0


### 断点恢复
bash train.sh place_empty_cup dp_test_demo 50 0 14 0 \
  checkpoints/place_empty_cup-dp_test_demo-50-0/500.ckpt

### 查看loss
ppython - <<'PY'
import json
import glob
import os

files = glob.glob("data/outputs/**/logs.json.txt", recursive=True)
logfile = max(files, key=os.path.getmtime)

print("log:", logfile)
print()
print(f"{'epoch':>6} {'train_loss':>14} {'val_loss':>14} {'action_mse':>14} {'lr':>12}")

with open(logfile) as f:
    for line in f:
        d = json.loads(line)

        # 每个 epoch 结束的记录才有 val_loss
        if "val_loss" in d:
            print(
                f"{int(d['epoch']):6d} "
                f"{d['train_loss']:14.6f} "
                f"{d['val_loss']:14.6f} "
                f"{d.get('train_action_mse_error', float('nan')):14.6f} "
                f"{d['lr']:12.3e}"
            )
PY

### 查看loss
    python - <<'PY'
    import json
    import glob
    import os
    import math
    import matplotlib.pyplot as plt

    # 自动找到最近一次训练日志
    files = glob.glob("data/outputs/**/logs.json.txt", recursive=True)
    logfile = max(files, key=os.path.getmtime)

    epochs = []
    train_loss = []
    val_loss = []
    action_mse_epoch = []
    action_mse = []
    lr = []

    with open(logfile) as f:
        for line in f:
            d = json.loads(line)

            # 每个 epoch 的汇总记录带 val_loss
            if "val_loss" in d:
                e = int(d["epoch"])
                epochs.append(e)
                train_loss.append(d["train_loss"])
                val_loss.append(d["val_loss"])
                lr.append(d["lr"])

                if "train_action_mse_error" in d:
                    v = d["train_action_mse_error"]
                    if not math.isnan(v):
                        action_mse_epoch.append(e)
                        action_mse.append(v)

    print("log:", logfile)

    # 1. Train / validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RoboTwin Diffusion Policy Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_loss_curve.png", dpi=200)
    plt.close()

    # 2. 对数坐标，更适合你这个跨度
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_loss, label="train_loss")
    plt.semilogy(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("RoboTwin DP Loss - Log Scale")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_loss_curve_log.png", dpi=200)
    plt.close()

    # 3. Action MSE
    plt.figure(figsize=(10, 6))
    plt.semilogy(action_mse_epoch, action_mse)
    plt.xlabel("Epoch")
    plt.ylabel("Train Action MSE")
    plt.title("DP Action Prediction MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_action_mse.png", dpi=200)
    plt.close()

    # 4. Learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("DP Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_lr_curve.png", dpi=200)
    plt.close()

    print("saved:")
    print("  dp_loss_curve.png")
    print("  dp_loss_curve_log.png")
    print("  dp_action_mse.png")
    print("  dp_lr_curve.png")
    PY

### 评估最终模型
conda activate robotwin_dp
cd ~/RoboTwin

CUDA_VISIBLE_DEVICES=0 python script/eval_policy.py \
  --config policy/DP/deploy_policy.yml \
  --overrides \
  --task_name place_empty_cup \
  --task_config dp_test_demo \
  --ckpt_setting dp_test_demo \
  --expert_data_num 50 \
  --seed 0 \
  --checkpoint_num 1000

### 测试
bash eval_double_env.sh \
    place_empty_cup \
    dp_test_demo \
    dp_test_demo \
    50 \
    0 \
    0 \
    robotwin_dp