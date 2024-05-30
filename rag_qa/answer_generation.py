"""Answer generation"""
import argparse
import glob
import os
from typing import Optional

import pandas as pd
import transformers
import yaml

from rag_qa.modelling.question_answer import (
    answer_question_flan,
    load_model_flan,
)
from rag_qa.modelling.semantic_search import context_ranking, load_model_ss

pd.set_option("display.max_colwidth", 255)

with open("config/config.yaml", encoding="utf-8") as cf_file:
    config = yaml.safe_load(cf_file.read())

vector_method = config["preprocessing"]["vector_simalirity_method"]
model_semantic = config["modelling"]["semantic_model"]
qna_model = config["modelling"]["qna_model"]
use_gpu = config["modelling"]["use_gpu"]
top_k_context = config["modelling"]["top_k_context"]
min_ans_length = config["modelling"]["min_ans_length"]
max_ans_length = config["modelling"]["max_ans_length"]
no_repeat_ngram_size = config["modelling"]["no_repeat_ngram_size"]


def generate_answers(
    df: pd.DataFrame,
    index_dir: str,
    model_ss: transformers.models.mpnet.modeling_mpnet.MPNetModel,
    tokenizer_ss: transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
    model_qa: transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
    tokenizer_qa: transformers.models.t5.tokenization_t5_fast.T5TokenizerFast,
) -> pd.DataFrame:
    """Do the semantic search and then generate answers from top-k-contexts

    Args:
        df (pd.DataFrame): pandas dataframe containing the questions to be answered
        index_dir (str): folder path containing the indexed data
        model_ss (transformers.models.mpnet.modeling_mpnet.MPNetModel): Sementic model for context search
        tokenizer_ss (transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast): Sementic
        tokenizer for context search
        model_qa (transformers.models.t5.modeling_t5.T5ForConditionalGeneration): QnA model for generating answer
        tokenizer_qa (transformers.models.t5.tokenization_t5_fast.T5TokenizerFast): QnA tokenizer for generating answer

    Returns:
       pandas dataframe with generated ans along with top-k-contexts and their references
    """
    context = []
    final_ans = []
    context_refs: dict = {}
    for con in range(1, (top_k_context + 1)):
        context_refs[f"ContextRef_{con}"] = []
    for question in df["Question"]:
        op_files = glob.glob(os.path.join(index_dir, "*/*.csv"))
        context_df = context_ranking(
            question, op_files, model_ss, tokenizer_ss, vector_method
        )
        # answer generated from top 5 contexts
        main_context = "\n".join(context_df["content"].values[0:top_k_context])
        # to store in csv with marking
        main_context_store = "\n------\n".join(
            context_df["content"].values[0:top_k_context]
        )
        context.append(main_context_store)
        # QA
        output = answer_question_flan(
            model_qa,
            tokenizer_qa,
            main_context,
            question,
            use_gpu,
            min_ans_length,
            max_ans_length,
            no_repeat_ngram_size,
        )
        final_ans.append(output)
        # adding context reference
        for con in range(0, top_k_context):
            temp = (
                str(context_df["doc_name"].values[con])
                + ";"
                + str(context_df["page"].values[con])
            )
            var_list = context_refs[f"ContextRef_{con+1}"]
            var_list.append(temp)
    df["Extracted Context"] = context
    df["Generated Answer"] = final_ans
    for con in range(0, top_k_context):
        df[f"ContextRef_{con+1}"] = context_refs[f"ContextRef_{con+1}"]

    return df


def arg_parser(args: Optional[list] = None) -> argparse.Namespace:
    """Parse the command line arguments

    Args:
        args (list, optional): list of command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: the parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="RAG QA tool")
    parser.add_argument("--filepath", help="Path to a file containing questions")
    parser.add_argument(
        "--index_dir", default="output", help="Directory containing tokenized data"
    )
    return parser.parse_args(args)


def main() -> None:
    """Generate answers to a list of provided questions"""

    args = arg_parser()
    filepath = args.filepath
    index_dir = args.index_dir

    df = pd.read_csv(filepath)

    model_ss, tokenizer_ss = load_model_ss(model_semantic)
    model_qa, tokenizer_qa = load_model_flan(qna_model, use_gpu)

    df = generate_answers(df, index_dir, model_ss, tokenizer_ss, model_qa, tokenizer_qa)

    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
