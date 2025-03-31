# -*- coding: utf-8 -*-
""" Run ULTRA experiments """
import os
import re
import subprocess
from typing import List

import click
from loguru import logger

CHECKPOINTS = ["ultra_3g.pth", "ultra_4g.pth", "ultra_50g.pth"]
FOLDER_M = os.path.expanduser("~/git/ULTRA/output/Ultra/")

def get_narrative_dataset_classes(start_with: str = "NarrativeInductiveDataset") -> List[str]:
    """
    Find all classes in ultra/datasets.py that start with NarrativeInductiveDataset.
    
    Returns:
        List[str]: List of NarrativeInductiveDataset class names
    """
    datasets_file = os.path.join("ultra", "datasets.py")

    narrative_classes = []
    class_pattern = re.compile(r"^class (" + start_with + r"[^\(]*)")

    with open(datasets_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = class_pattern.match(line.strip())
            if match and match.group(1) != start_with:
                narrative_classes.append(match.group(1))

    return narrative_classes

def get_narrative_dataset_versions(path):
    return [x for x in os.listdir(path) if x.startswith("kg_base")]


def get_command(mode, info):
    """ Run script/run.py command """
    info = {k: str(v) for k, v in info.items()}
    if mode == "zero-shot":
        command = f"""
        python -m torch.distributed.launch --nproc_per_node=2 \
            script/run.py -c config/inductive/inference.yaml \
                --dataset {info['dataset']} --epochs {info['epoch']} \
                    --bpe {info['bpe']} --gpus [0,1] --ckpt {info['ckpt']} \
                        --version {info['version']} 
        """

    elif mode == "fine-tune":
        command = f"""
        python -m torch.distributed.launch --nproc_per_node=2 \
            script/run.py -c narrative/config/fine_tune.yaml \
                --finetune --dataset {info['dataset']} --epochs {info['epoch']} \
                    --bpe {info['bpe']} --gpus [0,1] --ckpt {info['ckpt']} \
                        --batch_size {info['bs']} --version {info['version']}
        """

    else:
        command = f"""
        python -m torch.distributed.launch --nproc_per_node=2 \
            script/pretrain.py -c narrative/config/pretrain.yaml \
                --graphs [{info['dataset']}] --epochs {info['epoch']} \
                    --bpe {info['bpe']} --gpus [0,1] --batch_size {info['bs']} --version {info['version']}
        """

    return command


def cp_latest_log_file(m, v, mode, folder):
    """ Copy latest log file of model {m} to {folder} """
    if mode == "pretrain":
        curr_folder = os.path.join(FOLDER_M, "JointDataset")
    else:
        curr_folder = os.path.join(FOLDER_M, m)
    exp = sorted(os.listdir(curr_folder))[-1]
    cp_file = os.path.join(folder, v + '_log.txt')
    command = f"cp {os.path.join(curr_folder, exp, 'log.txt')} {cp_file}"
    subprocess.call(command, shell=True)


@click.command()
@click.argument("mode", type=click.Choice(["zero-shot", "fine-tune", "pretrain"]))
@click.argument("ckpt_p", type=click.Path(exists=True))
@click.argument("folder_out")
@click.argument("version_f")
@click.argument("dataset")
@click.option('--include_role/--no-include-role', is_flag=True, default=True,
              help="Whether to include roles in the scripts (much bigger files)")
def main(mode, ckpt_p, folder_out, version_f, dataset, include_role):
    """
    Execute evaluation runs for narrative KGs across different configurations.
    This function orchestrates the execution of experiments for narrative KGs
    by configuring different hyperparameters based on the specified mode (zero-shot,
    fine-tune, or pretrain). It handles directory creation and avoids re-running
    configurations that have already been completed.
    Parameters
    ----------
    mode : str
        The evaluation mode: 'zero-shot', 'fine-tune', or 'pretrain', which determines
        the hyperparameter sets to use.
    ckpt_p : str
        Path to the directory containing model checkpoints.
    folder_out : str
        Path where output logs and results will be saved.
    include_role : bool
        Whether to include Role1 narrative classes in the evaluation.
    Notes
    -----
    The function creates a directory structure based on mode and configuration parameters,
    runs evaluation commands using subprocess, and copies log files to appropriate locations.
    It skips configurations that have already been run (determined by the existence of log files).
    """

    versions = get_narrative_dataset_versions(version_f)
    if not include_role:
        versions = [d for d in versions if "role_1" not in d]
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    if mode == "zero-shot":
        epochs = [0]
        bpes = ["null"]
        batch_size = [8]
    elif mode == "fine-tune":
        epochs = [1, 3, 5]
        bpes = [100, 1000, 2000, 4000]
        batch_size = [16, 64]
    else:  # mode == "pretrain"
        epochs = [5, 10]
        bpes = [1000, 2000]
        batch_size = [16, 64]

    checkpoints = CHECKPOINTS if mode != "pretrain" else ["ckpt"]

    # Run one exp per ckpt+epochs+bpe
    for ckpt_n in checkpoints:
        for epoch in epochs:
            for bpe in bpes:
                for bs in batch_size:
                    ckpt_short = ckpt_n.split('.', maxsplit=1)[0]
                    fp = os.path.join(folder_out, mode,
                                      f"ckpt_{ckpt_short}_epochs_{epoch}_bpe_{bpe}_bs_{bs}")
                    if not os.path.exists(fp):
                        os.makedirs(fp)
                    for v in versions:
                        logger.info(f"Config: DATASET {dataset} | VERSION {v} | CKPT {ckpt_n} | " + \
                            f"EPOCH {epoch} | BPE {bpe}")
                        log_name = f"{v}_log.txt"
                        if not os.path.exists(os.path.join(fp, log_name)):
                            logger.info("Running config")
                            command = get_command(
                                mode, {"dataset": dataset, "epoch": epoch, "bpe": bpe,
                                       "ckpt": os.path.join(ckpt_p, ckpt_n), "bs": bs,
                                       "version": v})
                            subprocess.call(command, shell=True)
                            cp_latest_log_file(dataset, v, mode, fp)
                        else:
                            logger.info("Config already run")
                        logger.info("--------------------")



if __name__ == '__main__':
    main()
