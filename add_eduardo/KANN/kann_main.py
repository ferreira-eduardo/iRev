# KANN_main.py (PyTorch version)
import os
import time
import pickle
import logging
import math
import glob
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from KANN import KA  # your PyTorch KA implementation

logging.basicConfig(level="ERROR")

np.random.seed(2020)
torch.manual_seed(2020)
random_seed = 2020


# -------------------------
# Loss / Metrics
# -------------------------
def loss_function(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(pred, real)


@torch.no_grad()
def evaluate_metric_rmse(real_eval: torch.Tensor, pred_eval: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred_eval - real_eval) ** 2))


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    TF version:
      mask = (seq == 0) -> float32
      returns (B,1,1,S) for broadcasting
    Keep the same shape so your KA/Interacter mask adapter can consume it.
    """
    mask = (seq == 0).to(torch.float32)
    return mask[:, None, None, :]  # (B,1,1,S)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)


# -------------------------
# Noam / Transformer LR schedule (TF CustomSchedule equivalent)
# -------------------------
class NoamLR:
    """
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        step = float(self.step_num)
        lr = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def _latest_checkpoint(checkpoint_dir: str):
    """
    Finds the latest checkpoint by epoch number in filenames like: ckpt_epoch_12.pt
    """
    paths = glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.pt"))
    if not paths:
        return None, 0

    def epoch_from_name(p):
        m = re.search(r"ckpt_epoch_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    paths.sort(key=epoch_from_name)
    latest = paths[-1]
    return latest, epoch_from_name(latest)


def _save_checkpoint(checkpoint_dir: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, max_to_keep: int = 5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    # keep only last N
    ckpts = glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.pt"))
    if len(ckpts) > max_to_keep:
        def epoch_from_name(p):
            m = re.search(r"ckpt_epoch_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else 0
        ckpts.sort(key=epoch_from_name)
        for p in ckpts[:-max_to_keep]:
            try:
                os.remove(p)
            except OSError:
                pass
    return path


# -------------------------
# Train / Eval / Test
# -------------------------
def train_one_step(model, optimizer, lr_sched, device,
                   u_batch, i_batch, uid, iid, reuid, reiid, y_batch):
    model.train()

    inp_padding_mask = create_padding_mask(u_batch)
    tar_padding_mask = create_padding_mask(i_batch)

    optimizer.zero_grad(set_to_none=True)

    # Keep the same argument order as TF main:
    # KA(u_batch, i_batch, True, uid, iid, reuid, reiid, inp_padding_mask, tar_padding_mask)
    scores, _ = model(u_batch, i_batch, True, uid, iid, reuid, reiid, inp_padding_mask, tar_padding_mask)

    loss = loss_function(y_batch, scores)
    loss.backward()
    optimizer.step()
    lr = lr_sched.step()

    return loss.item(), lr


@torch.no_grad()
def evaluate(model, device,
             u_valid, i_valid, userid_valid, itemid_valid, reuid_eval, reiid_eval, y_valid):
    model.eval()
    inpv_padding_mask = create_padding_mask(u_valid)
    tarv_padding_mask = create_padding_mask(i_valid)

    scores, _ = model(
        u_valid, i_valid, False,
        userid_valid, itemid_valid, reuid_eval, reiid_eval,
        inpv_padding_mask, tarv_padding_mask
    )

    loss = loss_function(y_valid, scores).item()
    rmse = evaluate_metric_rmse(y_valid, scores).item()
    return loss, rmse


@torch.no_grad()
def test(model, device,
         u_test, i_test, userid_test, itemid_test, reuid_test, reiid_test, y_test):
    model.eval()
    inptest_padding_mask = create_padding_mask(u_test)
    tartest_padding_mask = create_padding_mask(i_test)

    scores, _ = model(
        u_test, i_test, False,
        userid_test, itemid_test, reuid_test, reiid_test,
        inptest_padding_mask, tartest_padding_mask
    )

    loss = loss_function(y_test, scores).item()
    rmse = evaluate_metric_rmse(y_test, scores).item()
    return loss, rmse


def _pack_batch(data_batch, u_text, i_text, device):
    """
    TF code stores each field as something like uid[i] = [user_id].
    It indexes reviews with uid[i][0]. We replicate that.
    """
    uid, iid, reuid, reiid, y = zip(*data_batch)

    u_list, i_list = [], []
    uid_ids, iid_ids = [], []
    reuid_ids, reiid_ids = [], []

    for j in range(len(uid)):
        u_id = int(uid[j][0])
        i_id = int(iid[j][0])
        uid_ids.append(u_id)
        iid_ids.append(i_id)

        # keep these if your KA uses them
        reuid_ids.append(int(reuid[j][0]) if hasattr(reuid[j], "__len__") else int(reuid[j]))
        reiid_ids.append(int(reiid[j][0]) if hasattr(reiid[j], "__len__") else int(reiid[j]))

        u_review = np.reshape(u_text[u_id], (-1))
        i_review = np.reshape(i_text[i_id], (-1))
        u_list.append(u_review)
        i_list.append(i_review)

    u_batch = torch.as_tensor(np.array(u_list), dtype=torch.long, device=device)
    i_batch = torch.as_tensor(np.array(i_list), dtype=torch.long, device=device)

    uid_t = torch.as_tensor(uid_ids, dtype=torch.long, device=device)
    iid_t = torch.as_tensor(iid_ids, dtype=torch.long, device=device)
    reuid_t = torch.as_tensor(reuid_ids, dtype=torch.long, device=device)
    reiid_t = torch.as_tensor(reiid_ids, dtype=torch.long, device=device)

    y_batch = torch.as_tensor(np.array(y), dtype=torch.float32, device=device).view(-1, 1)

    return u_batch, i_batch, uid_t, iid_t, reuid_t, reiid_t, y_batch


def main():
    output_dir = "../output/amazon/10core"
    data_dir = "../data/amazon/10core"

    checkpoint_root = os.path.join(output_dir, "checkpoints")
    log_root = os.path.join(output_dir, "logs")

    print("Loading data...")

    train_data_dir = os.path.join(data_dir, "amazon.train")
    valid_data_dir = os.path.join(data_dir, "amazon.valid")
    test_data_dir = os.path.join(data_dir, "amazon.test")
    utext_dir = os.path.join(data_dir, "user_review")
    itext_dir = os.path.join(data_dir, "item_review")
    para_dir = os.path.join(data_dir, "amazon.para")

    with open(train_data_dir, "rb") as f:
        train_data = pickle.load(f)
    with open(valid_data_dir, "rb") as f:
        valid_data = pickle.load(f)
    with open(test_data_dir, "rb") as f:
        test_data = pickle.load(f)
    with open(utext_dir, "rb") as f:
        u_text = pickle.load(f)
    with open(itext_dir, "rb") as f:
        i_text = pickle.load(f)
    with open(para_dir, "rb") as f:
        para = pickle.load(f)

    user_num = para["user_num"]
    item_num = para["item_num"]

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    test_data = np.array(test_data)

    # Initialize KANN (same hyperparams)
    num_layers = 2
    d_model = 50
    num_heads = 4
    dff = 1024
    batch_size = 64
    EPOCHS = 38

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KA(user_num, item_num, num_layers, d_model, num_heads, dff).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # will be overridden by Noam schedule at step 1
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    lr_sched = NoamLR(optimizer, d_model=d_model, warmup_steps=4000)

    # Run id (keep same naming style)
    run_id = f"wordemb_{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_head"
    checkpoint_path = os.path.join(checkpoint_root, run_id)
    log_dir = os.path.join(log_root, run_id)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # Restore latest checkpoint if exists
    latest_ckpt, last_epoch = _latest_checkpoint(checkpoint_path)
    if latest_ckpt is not None:
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        last_epoch = int(ckpt.get("epoch", last_epoch))
        print(f"have read the newest checkpoint, and the model has trained {last_epoch} epochs.")
    else:
        last_epoch = 0
        print("haven't found checkpoint, training from start.")

    print(f"KANN has trained {last_epoch} epochs.")
    print(f"Left epochs: {max(0, EPOCHS - last_epoch)}")

    data_size_train = len(train_data)
    data_size_valid = len(valid_data)
    data_size_test = len(test_data)

    ll = int(data_size_train / batch_size)
    ll_valid = int(data_size_valid / batch_size)
    ll_test = int(data_size_test / batch_size)

    global_step = last_epoch * max(1, ll)

    for epoch in range(last_epoch, EPOCHS):
        start = time.time()

        shuffle_indices = np.random.permutation(np.arange(data_size_train))
        shuffled_data = train_data[shuffle_indices]

        train_loss_meter = AverageMeter()
        valid_loss_meter = AverageMeter()
        valid_rmse_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        test_rmse_meter = AverageMeter()

        # ---- Train ----
        for batch_num in range(ll):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size_train)
            data_train = shuffled_data[start_index:end_index]

            u_batch, i_batch, uid, iid, reuid, reiid, y_batch = _pack_batch(data_train, u_text, i_text, device)

            loss_val, lr = train_one_step(model, optimizer, lr_sched, device,
                                          u_batch, i_batch, uid, iid, reuid, reiid, y_batch)
            train_loss_meter.update(loss_val, n=u_batch.size(0))

            global_step += 1
            if batch_num % 200 == 0:
                writer.add_scalar("lr", lr, global_step)

        # ---- Valid ----
        print("\nEvaluation:")
        print("the batch number is: ", batch_num)

        for valid_batch in range(ll_valid):
            start_index = valid_batch * batch_size
            end_index = min((valid_batch + 1) * batch_size, data_size_valid)
            data_valid = valid_data[start_index:end_index]

            u_valid, i_valid, uid_v, iid_v, reuid_v, reiid_v, y_valid = _pack_batch(data_valid, u_text, i_text, device)

            v_loss, v_rmse = evaluate(model, device, u_valid, i_valid, uid_v, iid_v, reuid_v, reiid_v, y_valid)
            valid_loss_meter.update(v_loss, n=u_valid.size(0))
            valid_rmse_meter.update(v_rmse, n=u_valid.size(0))

        # ---- Test ----
        print("\ntest:")
        for test_batch in range(ll_test):
            start_index = test_batch * batch_size
            end_index = min((test_batch + 1) * batch_size, data_size_test)
            data_t = test_data[start_index:end_index]

            u_test, i_test, uid_t, iid_t, reuid_t, reiid_t, y_test = _pack_batch(data_t, u_text, i_text, device)

            t_loss, t_rmse = test(model, device, u_test, i_test, uid_t, iid_t, reuid_t, reiid_t, y_test)
            test_loss_meter.update(t_loss, n=u_test.size(0))
            test_rmse_meter.update(t_rmse, n=u_test.size(0))

        # ---- Save checkpoint each epoch ----
        ckpt_save_path = _save_checkpoint(checkpoint_path, epoch + 1, model, optimizer, max_to_keep=5)
        print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

        # ---- TensorBoard ----
        writer.add_scalar("train_loss", train_loss_meter.avg, epoch + 1)
        writer.add_scalar("valid_loss", valid_loss_meter.avg, epoch + 1)
        writer.add_scalar("valid_rmse", valid_rmse_meter.avg, epoch + 1)
        writer.add_scalar("test_loss", test_loss_meter.avg, epoch + 1)
        writer.add_scalar("test_rmse", test_rmse_meter.avg, epoch + 1)

        print(f"Epoch {epoch + 1} Train Loss {train_loss_meter.avg:.4f}")
        print(f"Epoch {epoch + 1} Valid Loss {valid_loss_meter.avg:.4f}")
        print(f"Epoch {epoch + 1} Valid RMSE {valid_rmse_meter.avg:.4f}")
        print(f"Epoch {epoch + 1} Test Loss {test_loss_meter.avg:.4f}")
        print(f"Epoch {epoch + 1} Test RMSE {test_rmse_meter.avg:.4f}")
        print(f"Time taken for 1 epoch: {time.time() - start} secs\n")

    writer.close()


if __name__ == "__main__":
    main()
