"""Compare the answers with the ground truth and store the results in the specified output directory"""
import argparse
from typing import Optional

import pandas as pd
import yaml

from rag_qa.model_validation.model_validation import answer_scoring
from rag_qa.modelling.semantic_search import load_model_ss


def arg_parser(args: Optional[list] = None) -> argparse.Namespace:
    """Parse the command line arguments

    Args:
        args (list, optional): list of command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: the parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="RAG QA tool")
    parser.add_argument(
        "--filepath", help="Path to a file containing context, answers and ground truth"
    )
    return parser.parse_args(args)


def main() -> None:
    """
    Parse the command line arguments then score the generated answers against the ground truth.

    Returns:
        None
    """
    args = arg_parser()
    filepath = args.filepath

    with open("config/config.yaml", encoding="utf-8") as cf_file:
        config = yaml.safe_load(cf_file.read())

    model_semantic = config["modelling"]["semantic_model"]

    df = pd.read_csv(filepath)

    model, tokenizer = load_model_ss(model_semantic)

    df = answer_scoring(df, model, tokenizer)

    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
