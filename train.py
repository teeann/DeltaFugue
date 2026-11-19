import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except Exception as _wandb_exc:  # noqa: BLE001
    # Logging is optional -- results are also printed to stdout and written to
    # results/ -- so a broken or missing wandb install must not stop a run.
    print(f"WARNING: wandb unavailable ({type(_wandb_exc).__name__}: {_wandb_exc}); "
          f"continuing without experiment logging.")

    class _NoOpWandb:
        class _Config:
            def update(self, *args, **kwargs):
                pass

        config = _Config()

        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            pass

        def finish(self, *args, **kwargs):
            pass

    wandb = _NoOpWandb()

from data_dir.dataloaders import create_fl_dataloaders
from models.lr_scheduler import LinearWarmupCosineAnnealing


def set_seed(seed=42):
    """Seeds Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _memory_modules():
    from fla.layers import DeltaFugue, GatedDeltaFugue
    from models.stack_rnn import StackRNN
    from models.tape_rnn import TapeRNN
    return (DeltaFugue, GatedDeltaFugue, TapeRNN), (StackRNN,)


def set_eval_memory_size(model: nn.Module, tape_size: int, stack_size: int):
    """Grows the tape / stack of every tape or stack layer for length-generalization eval."""
    tape_types, stack_types = _memory_modules()
    for module in model.modules():
        if isinstance(module, tape_types):
            module.set_tape_size(tape_size)
        elif isinstance(module, stack_types):
            module.set_stack_size(stack_size)


def reset_memory_size(model: nn.Module):
    """Restores the training-time tape / stack size after eval."""
    tape_types, stack_types = _memory_modules()
    for module in model.modules():
        if isinstance(module, tape_types):
            module.reset_tape_size()
        elif isinstance(module, stack_types):
            module.reset_stack_size()


def evaluate(model, dataloader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for X_val, y_val, mask_val in dataloader:
            X_val, y_val, mask_val = X_val.to(device), y_val.to(device), mask_val.to(device)
            outputs_val = model(X_val)
            correct += (outputs_val[mask_val].argmax(dim=-1) == y_val[mask_val].flatten()).sum().item()
            total += y_val[mask_val].flatten().size(0)
    return correct / total if total > 0 else 0.0


def train_model(
    config,
    data_dim,
    label_dim,
    task,
    model_name,
    model,
    dataloader,
    num_steps,
    print_steps,
    learning_rate,
    device,
    early_stop_threshold=1.0,
    weight_decay_embedding=0.0,
    weight_decay_others=1e-2,
    warmup_fraction=0.15,
    final_lr=1e-5,
    val_tape_size=256,
    val_stack_size=256,
):
    """
    Trains for `num_steps` steps. Every `print_steps` steps the model is evaluated
    on the (longer) validation split, with tape / stack layers resized to
    `val_tape_size` / `val_stack_size`; training stops early once validation
    accuracy reaches `early_stop_threshold`.

    Returns:
        model, steps (list[int]), val_accs (list[float]), early_stop (bool)
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"{task}_{model_name}_{time_str}_seed{config.get('seed', 0)}_pid{os.getpid()}"

    model.to(device)
    embedding_params = [p for n, p in model.named_parameters() if "embedding" in n]
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    optimizer = optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": weight_decay_embedding},
            {"params": other_params, "weight_decay": weight_decay_others},
        ],
        lr=learning_rate,
    )

    # `warmup_steps` (absolute) overrides `warmup_fraction`. The schedule
    # depends on `num_steps`, so two runs with different `num_steps` differ
    # from the first step past warmup on, even with the same seed.
    warmup_steps = config.get("warmup_steps", int(warmup_fraction * num_steps))
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer, warmup_steps, num_steps, learning_rate, final_lr
    )

    criterion = nn.CrossEntropyLoss()
    step = 0
    total_loss = 0
    best_val_acc = 0
    best_step = 0
    total_train_acc = 0
    steps = []
    val_accs = []
    start_time = time.time()
    save_checkpoint = config.get("save_checkpoint", False)

    model.train()
    while step < num_steps:
        for X, y, mask in dataloader["train"]:
            optimizer.zero_grad()

            X, y, mask = X.to(device), y.to(device), mask.to(device)
            outputs = model(X)
            loss = criterion(outputs[mask], y[mask].flatten())

            with torch.no_grad():
                masked_targets = y[mask].flatten()
                if masked_targets.numel() > 0:
                    preds = outputs[mask].argmax(dim=-1)
                    batch_acc = (preds == masked_targets).float().mean().item()
                else:
                    batch_acc = 0.0

            loss.backward()
            model.mask_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            total_train_acc += batch_acc

            if step % print_steps == 0:
                set_eval_memory_size(model, val_tape_size, val_stack_size)
                model.eval()
                end_time = time.time()
                accuracy = evaluate(model, dataloader["val"], device)

                # At step 0 only one batch has been accumulated.
                if step == 0 and print_steps > 1:
                    total_loss *= print_steps
                    total_train_acc *= print_steps

                avg_loss = total_loss / print_steps
                avg_train_acc = total_train_acc / print_steps

                print(
                    f"Step: {step}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Train Acc: {avg_train_acc:.4f}, "
                    f"Validation Acc: {accuracy:.4f}, "
                    f"Time: {end_time - start_time:.2f} sec"
                )

                wandb.log(
                    {
                        "step": step,
                        "train/loss": avg_loss,
                        "train/accuracy": avg_train_acc,
                        "val/accuracy": accuracy,
                        "val/best_accuracy": max(best_val_acc, accuracy),
                        "time/elapsed": end_time - start_time,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }
                )

                steps.append(step)
                val_accs.append(accuracy)

                if accuracy > best_val_acc:
                    best_val_acc, best_step = accuracy, step
                    if save_checkpoint:
                        checkpoint_path = os.path.join("checkpoints", f"checkpoint_{run_tag}.pt")
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "config": config,
                                "data_dim": data_dim,
                                "label_dim": label_dim,
                                "model_name": model_name,
                                "step": step,
                                "val_acc": accuracy,
                            },
                            checkpoint_path,
                        )

                early_stop = accuracy >= early_stop_threshold

                with open(os.path.join("results", f"results_{run_tag}.json"), "w") as f:
                    json.dump(
                        {
                            "config": config,
                            "steps": steps,
                            "val_accs": val_accs,
                            "best_val_acc": best_val_acc,
                            "best_step": best_step,
                            "early_stop": early_stop,
                        },
                        f,
                        indent=2,
                    )

                start_time = time.time()
                total_loss = 0
                total_train_acc = 0

                reset_memory_size(model)
                model.train()

                if early_stop:
                    return model, steps, val_accs, True

            if step > num_steps:
                break

            step += 1

    return model, steps, val_accs, False


def build_model(config, model_name, data_dim, label_dim):
    num_blocks = config["num_blocks"]
    model_dim = config["model_dim"]
    num_heads = config.get("num_heads", 4)
    use_glu = config.get("use_glu", False)
    second_embedding = config.get("second_embedding", False)
    gated = config.get("gated", True)
    dropout_rate = config.get("dropout_rate", 0.01)
    rank = config.get("rank", 0)
    tape_size = config.get("tape_size", None)
    pointer_gamma_init = config.get("pointer_gamma_init", 2.0)  # tape pointer sharpness init (softplus)

    if model_name in ["deltanet", "gateddeltanet", "rwkv7", "rwkv6", "deltaproduct"]:
        from models.fla import StackedBlock

        return StackedBlock(
            layer_type=model_name,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
            rank=rank,
            gated=gated,
        )
    if model_name in ["deltafugue", "gated_deltafugue"]:
        from models.fla import StackedBlock

        return StackedBlock(
            num_heads=num_heads,
            layer_type=model_name,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
            rank=rank,
            gated=gated,
            tape_size=tape_size,
            pointer_gamma_init=pointer_gamma_init,
        )
    if model_name == "deltanet2":
        from models.deltanet2 import StackedBlock

        return StackedBlock(
            num_heads=num_heads,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )
    if model_name == "mamba":
        from models.mamba import StackedMamba

        return StackedMamba(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )
    if model_name == "lstm":
        from models.lstm import LSTM

        return LSTM(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
        )
    if model_name == "stack_rnn":
        from models.stack_rnn import StackRNN

        return StackRNN(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
            stack_size=config.get("stack_size", 32),
        )
    if model_name == "tape_rnn":
        from models.tape_rnn import TapeRNN

        return TapeRNN(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
            tape_size=tape_size if tape_size is not None else 64,
            # Number of delimiter tokens (e.g. SEP) to exclude when deriving the
            # jump length from the input; see models/tape_rnn.py.
            input_length_offset=config.get("input_length_offset", 0),
        )
    if model_name == "transformer_causal":
        from models.transformer import CausalTransformer

        return CausalTransformer(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
        )
    raise ValueError(f"Model not recognized: {model_name}")


def run_experiment(config):
    """
    Builds the train / validation dataloaders and the model from `config`,
    trains, and returns (model, steps, val_accs, early_stop).

    The default protocol is length generalization: train on inputs of length
    3-60, validate on lengths 60-256.
    """
    seed = config.get("seed", 0) + config.get("run", 0)
    set_seed(seed)

    task = config["task"]
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    print_steps = config["print_steps"]
    learning_rate = config["learning_rate"]

    wandb_kwargs = dict(
        project=config.get("wandb_project", "deltafugue"),
        config=config,
        name=f"{task}_{model_name}_lr{learning_rate}_seed{seed}",
        reinit=True,
    )
    try:
        wandb.init(**wandb_kwargs)
    except Exception as exc:  # noqa: BLE001 -- e.g. no API key configured; logging is optional
        print(f"WARNING: wandb.init failed ({type(exc).__name__}: {exc}); continuing with wandb disabled.")
        wandb.init(mode="disabled", **wandb_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, _, data_dim, label_dim = create_fl_dataloaders(
        task,
        num_samples=config.get("train_num_samples", 25600000),
        batch_size=batch_size,
        min_length=config.get("train_min_length", 3),
        max_length=config.get("train_max_length", 60),
        padding_length=config.get("train_padding_length", 128),
        train_split=1.0,
        seed=seed,
    )
    val_dataloader, _, _, _ = create_fl_dataloaders(
        task,
        num_samples=config.get("val_num_samples", 8192),
        batch_size=batch_size,
        min_length=config.get("eval_min_length", 60),
        max_length=config.get("eval_max_length", 256),
        padding_length=config.get("val_padding_length", 512 + 4),  # room for separator / EOS
        train_split=1.0,
        seed=2 * seed,
    )
    dataloader = {"train": train_dataloader, "val": val_dataloader}

    model = build_model(config, model_name, data_dim, label_dim)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")
    wandb.config.update({"total_params": pytorch_total_params})

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    model, steps, val_accs, early_stop = train_model(
        config=config,
        data_dim=data_dim,
        label_dim=label_dim,
        task=task,
        model_name=model_name,
        model=model,
        dataloader=dataloader,
        num_steps=num_steps,
        print_steps=print_steps,
        learning_rate=learning_rate,
        early_stop_threshold=config.get("early_stop_threshold", 1.0),
        device=device,
        weight_decay_others=config.get("wd", 1e-2),
        warmup_fraction=config.get("warmup_fraction", 0.15),
        val_tape_size=config.get("val_tape_size", 256),
        val_stack_size=config.get("val_stack_size", 256),
    )

    if early_stop:
        print("Early stop triggered! Finished before reaching num_steps.")
    else:
        print("Training complete. Did not trigger early stopping.")

    wandb.finish()
    return model, steps, val_accs, early_stop
