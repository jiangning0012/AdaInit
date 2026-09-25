# -*- coding: utf-8 -*-
import os
import time
import json
from typing import Any

import torch

import ttab.utils.file_io as file_io


def resolve_resume_checkpoint(path: str) -> str:
    """Resolve either a checkpoint file or a run directory to a checkpoint file."""
    resolved_path = os.path.abspath(path)
    if os.path.isdir(resolved_path):
        marker_path = os.path.join(resolved_path, "latest_checkpoint.json")
        if not os.path.exists(marker_path):
            raise FileNotFoundError(
                f"No latest_checkpoint.json found in resume directory {resolved_path}."
            )
        with open(marker_path, "r") as fp:
            marker = json.load(fp)
        resolved_path = os.path.join(resolved_path, marker["filename"])
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Recovery checkpoint does not exist: {resolved_path}")
    return resolved_path


def init_checkpoint(conf: Any):
    if getattr(conf, "resume_checkpoint", None):
        conf.resume_checkpoint = resolve_resume_checkpoint(conf.resume_checkpoint)
        conf.checkpoint_path = os.path.dirname(conf.resume_checkpoint)
        return conf.checkpoint_path

    # init checkpoint dir.
    conf.checkpoint_path = os.path.join(
        conf.root_path,
        conf.model_name,
        conf.job_name,
        # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
        f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
    )

    # if the directory does not exists, create them.
    file_io.build_dirs(conf.checkpoint_path)
    return conf.checkpoint_path


def save_recovery_checkpoint(state: dict, folder_path: str, completed_domains: int):
    """Atomically save a domain-boundary recovery checkpoint and latest marker."""
    filename = f"ctta_checkpoint_domain_{completed_domains:02d}.pt"
    checkpoint_path = os.path.join(folder_path, filename)
    temporary_path = checkpoint_path + ".tmp"
    torch.save(state, temporary_path)
    with open(temporary_path, "rb") as fp:
        os.fsync(fp.fileno())
    os.replace(temporary_path, checkpoint_path)

    marker_path = os.path.join(folder_path, "latest_checkpoint.json")
    marker_temporary_path = marker_path + ".tmp"
    with open(marker_temporary_path, "w") as fp:
        json.dump(
            {"filename": filename, "completed_domains": completed_domains},
            fp,
            indent=2,
        )
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(marker_temporary_path, marker_path)
    return checkpoint_path


def load_recovery_checkpoint(path: str):
    resolved_path = resolve_resume_checkpoint(path)
    return resolved_path, torch.load(resolved_path, map_location="cpu")


def save_arguments(conf: Any, force: bool = False):
    # save the configure file to the checkpoint.
    path = os.path.join(conf.checkpoint_path, "arguments.json")

    if force or not os.path.exists(path):
        with open(path, "w") as fp:
            json.dump(
                dict(
                    [
                        (k, v)
                        for k, v in conf.__dict__.items()
                        if file_io.is_jsonable(v) and type(v) is not torch.Tensor
                    ]
                ),
                fp,
                indent=" ",
            )
