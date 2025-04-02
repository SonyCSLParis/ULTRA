# -*- coding: utf-8 -*-
"""
Retrieve results from fine-tuned models
"""
import os
import sys
import pprint
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from script.run import test
from ultra import util
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

def load_vocab(config_dataset):
    path = os.path.join(
        os.path.expanduser(config_dataset["root"]),
        config_dataset["class"],
        config_dataset["version"],
        "vocab")
    return torch.load(path)

def filter_edges_by_predicate(data, predicate_id):
    """Filters the graph edges and returns a subset of the data with predicate ID equal to `predicate_id`."""
    
    # Step 1: Create a mask for edges where edge_type == predicate_id
    mask = data.target_edge_type == predicate_id

    # Step 1.5: Find the indices where the mask is True
    masked_indices = mask.nonzero(as_tuple=True)[0]
    # Apply the mask to get the filtered edge_index and edge_type
    filtered_edge_index = data.target_edge_index[:, masked_indices]

    filtered_edge_type = data.target_edge_type[masked_indices]
    print(filtered_edge_index.shape, filtered_edge_type.shape)
    
    # Step 3: Return the filtered Data object using subgraph
    filtered_data = Data(
        target_edge_index=filtered_edge_index,
        target_edge_type=filtered_edge_type,
        num_nodes=data.num_nodes,
        num_relations=data.num_relations,
        edge_index=data.edge_index,
        edge_type=data.edge_type,
    )
    filtered_data = build_relation_graph(filtered_data)
    
    return filtered_data


if __name__ == '__main__':
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    test_data = test_data.to(device)
    valid_data = valid_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    model = model.to(device)

    if any(x in cfg.dataset['class'] for x in ["ILPC", "Ingram", "Narrative"]):
        # add inference, valid, test as the validation and test filtering graphs
        full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
        full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
        test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
        val_filtered_data = test_filtered_data
    else:
        # test filtering graph: inference edges + test edges
        full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
        full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
        test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

        # validation filtering graph: train edges + validation edges
        val_filtered_data = Data(
            edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
            edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
        )
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    vocab = load_vocab(cfg.dataset)
    # Get the name from vocab for each unique relation type in test data
    for id_ in test_data.target_edge_type.unique():
        # Find the key (name) in inv_rel_vocab for which the value is id_
        name = next((k for k, v in vocab["test"]["inv_rel_vocab"].items() if v == id_), f"Unknown relation {id_}")
        logger.info(f"Testing {name} ({id_})")
        filtered_data = filter_edges_by_predicate(test_data, id_)
        print(filtered_data.target_edge_index)
        print(filtered_data.target_edge_type)
        #Run test only on the filtered data
        test(cfg, model, filtered_data, filtered_data=test_filtered_data, device=device, logger=logger)
        logger.info(f"Testing {name} ({id_}) finished\n======")
    
    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)