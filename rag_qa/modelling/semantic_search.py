"""Semantic search functions"""
import os
import re
from typing import Tuple

import pandas as pd
import transformers
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from rag_qa.preprocessing.preprocessing import get_embeddings


def load_model_ss(
    model_checkpoint: str = "navteca/multi-qa-mpnet-base-cos-v1",
) -> Tuple[
    transformers.models.mpnet.modeling_mpnet.MPNetModel,
    transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
]:
    """Load in a pre-trained Huggingface model and tokenizer for semantic search

    Args:
        model_checkpoint (str, optional): reference to the pre-trained Huggingface model to load.
        Defaults to "sentence-transformers/multi-qa-mpnet-base-dot-v1".

    Returns:
        Tuple[
            transformers.models.mpnet.modeling_mpnet.MPNetModel,
            transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
            ]: pre-trained Huggingface model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    return model, tokenizer


def context_ranking(
    question: str,
    op_files: list,
    model: transformers.models.mpnet.modeling_mpnet.MPNetModel,
    tokenizer: transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
    vector_method: str = "FlatL2",
) -> pd.DataFrame:
    """_summary_

    Args:
        question (str): _description_
        op_files (list): _description_
        model (transformers.models.mpnet.modeling_mpnet.MPNetModel): _description_
        tokenizer (transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast): _description_
        vector_method: Type of vector method selected for indexing, FlatL2, FlatIP, FlatIVF
    Returns:
        pd.DataFrame: _description_
    """
    question_embedding = (
        get_embeddings([question], model, tokenizer).cpu().detach().numpy()
    )
    final_df = pd.DataFrame()
    for file in op_files:
        temp_df = pd.read_csv(file)
        content_dataset = Dataset.from_pandas(temp_df)
        name = re.sub(".csv", "", file)
        indx_fl = name + "_my_index.faiss"
        content_dataset.load_faiss_index("embeddings", indx_fl)
        scores, samples = content_dataset.get_nearest_examples(
            "embeddings", question_embedding, k=10
        )
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df["doc_name"] = (
            os.path.dirname(name).split("\\")[-1]
            + ";"
            + os.path.basename(name)
            + ".pdf"
        )
        final_df = pd.concat([final_df, samples_df], axis=0)
    final_df["temp_cont"] = [i.lower() for i in final_df["content"]]
    final_df.drop_duplicates(subset=["temp_cont"], keep="first", inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    if vector_method == "Cosine":
        final_df.sort_values("scores", ascending=False, inplace=True)
    else:
        final_df.sort_values("scores", ascending=True, inplace=True)

    return final_df
