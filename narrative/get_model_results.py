# -*- coding: utf-8 -*-
"""
Retrieve results from fine-tuned models
"""
import os
import re
import ast
import subprocess
import click
import pandas as pd

def concat_collapse_dict(l_dict):
    concat = l_dict[0]
    for x in l_dict[1:]:
        concat.update(x)
    res = {}
    for k1, v1 in concat.items():
        if isinstance(v1, dict):
            res.update({f"{k1}_{k2}": v2 for k2, v2 in v1.items()})
        else:
            res[k1] = v1
    return res

def get_config(lg_p):
    config = []
    with open(lg_p, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line[0].isdigit() and i <= 2:
                config.append(line[11:].strip())
            elif not line[0].isdigit():
                config.append(line.strip())
            else:
                break
    return ast.literal_eval(" ".join(config[2:]))
        
def parse_ultra_log(log_file_path):
    """
    Parse ULTRA model evaluation logs and extract metrics for validation and test datasets.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing metrics for validation and test datasets
    """
    
    results = {'valid': {}, 'test': {}}
    
    # Regular expressions for parsing
    dataset_pattern = r"Evaluate on (valid|test)"
    metric_pattern = r"(\w+@?\d*): ([\d\.]+)"
    
    current_dataset = None
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Check if this line indicates which dataset we're evaluating
                dataset_match = re.search(dataset_pattern, line)
                if dataset_match:
                    current_dataset = dataset_match.group(1)
                    continue
                
                # If we've identified a dataset, look for metrics
                if current_dataset:
                    metric_match = re.search(metric_pattern, line)
                    if metric_match:
                        metric_name = metric_match.group(1)
                        metric_value = float(metric_match.group(2))
                        results[current_dataset][metric_name] = metric_value
    
        return results
    
    except:
        return {}


def get_info(dataset):
    res = {}
    for name in ["prop", "subevent", "role", "causation"]:
        res[name] = 1 if f"{name.capitalize()}1" in dataset else 0
    res["syntax"] = dataset.split("Syntax")[1]
    return res

@click.command()
@click.argument("folder", type=click.Path(exists=True))
def main(folder):
    """
    Processes model results in a folder and saves them to a CSV file.
    This function iterates through files in the specified folder that start with 'NarrativeDataset',
    parses their log files to extract results, configuration details and dataset information,
    then combines these into a DataFrame and saves it as 'results.csv'.
    Args:
        folder (str): Path to the directory containing model results directories.
    Returns:
        None: The function saves the results to a CSV file but doesn't return any value.
    Side effects:
        Creates a 'results.csv' file in the specified folder containing 
        the consolidated model results.
    """

    data = []
    models = [x for x in os.listdir(folder) if x.startswith("NarrativeDataset")]
    for m in models:
        cp_file = os.path.join(folder, m + '_log.txt')
        res = parse_ultra_log(cp_file)
        config = get_config(cp_file)
        d_info = get_info(dataset=m)
        data.append(concat_collapse_dict([d_info, config, res]))

    pd.DataFrame(data).to_csv(os.path.join(folder, "results.csv"))


if __name__ == '__main__':
    # Example usage:
    # python narrative/get_model_results.py \
    #     narrative/experiments/zero-shot/ckpt_ultra_3g_epochs_0_bpe_null_bs_8
    main()
