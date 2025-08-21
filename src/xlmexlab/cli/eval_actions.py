import json
from typing import Any, Dict
from xlmexlab.parser import KeywordSearching

import click
import pandas as pd

from xlmexlab.evaluator import Evaluator


@click.command()
@click.argument("reference_dataset_path", type=str)
@click.argument("evaluated_dataset_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--action_similarity_threshold",
    default=0.8,
    help="Minimum threshold value to consider two actions as equals",
)
@click.option(
    "--chemical_similarity_threshold",
    default=0.7,
    help="Minimum threshold value to consider two actions as equals",
)
def eval_actions(
    reference_dataset_path: str,
    evaluated_dataset_path: str,
    output_file_path: str,
    action_similarity_threshold: float,
    chemical_similarity_threshold: float,
) -> None:
    
    evaluator = Evaluator(reference_dataset_path=reference_dataset_path)
    #chemicals: Dict[str, Any] = evaluator.evaluate_chemicals(evaluated_dataset_path, threshold=chemical_similarity_threshold)
    metadata: Dict[str, Any] = {
        "action_threshold": action_similarity_threshold,
        "chemical_threshold": chemical_similarity_threshold,
    }
    chemicals: Dict[str, Any] = evaluator.evaluate_chemicals(evaluated_dataset_path, threshold=chemical_similarity_threshold)
    chemicals_dict: Dict[str, Any] = {
        "chemical_precision": chemicals["precision"],
        "chemical_recall": chemicals["recall"],
        "chemical_f-score": chemicals["f-score"]
    }
    results: Dict[str, Any] = {**{"sequence": evaluator.evaluate_actions_order(evaluated_dataset_path)["accuracy"]}, **evaluator.evaluate_actions(evaluated_dataset_path, threshold=action_similarity_threshold), **chemicals_dict, **metadata}
    print(results)
    df = pd.DataFrame(results, index=[0])
    df.to_excel(output_file_path, index=False,)

def main():
    eval_actions()


if __name__ == "__main__":
    main()
