# pylint: disable=missing-module-docstring, missing-function-docstring
import glob
import os
from unittest.mock import Mock, patch

import pandas as pd

from dpp_helpline_qa.modelling.semantic_search import context_ranking, load_model_ss


@patch("dpp_helpline_qa.modelling.semantic_search.AutoModel")
@patch("dpp_helpline_qa.modelling.semantic_search.AutoTokenizer")
def test_load_model_ss(tokenizer_mock: Mock, model_mock: Mock) -> None:
    model_checkpoint = "test_model_checkpoint"

    load_model_ss(model_checkpoint)

    tokenizer_mock.from_pretrained.assert_called_with(model_checkpoint)
    model_mock.from_pretrained.assert_called_with(model_checkpoint)


def test_context_ranking() -> None:
    question = "What is Inventory?"
    op_files = glob.glob(
        os.path.join("tests", "test_data", "index", "test_data", "*.csv")
    )
    model, tokenizer = load_model_ss()

    final_df = context_ranking(question, op_files, model, tokenizer, "FlatL2")

    assert final_df.columns.tolist() == [
        "content",
        "page",
        "embeddings",
        "scores",
        "doc_name",
        "temp_cont",
    ]
    assert len(final_df) == 10 * len(op_files)

    assert len(final_df["doc_name"].unique()) == len(op_files)

    final_df_sorted = final_df.sort_values("scores", ascending=True)

    pd.testing.assert_frame_equal(final_df, final_df_sorted)

    final_df = context_ranking(question, op_files, model, tokenizer, "Cosine")

    final_df_sorted = final_df.sort_values("scores", ascending=False)

    pd.testing.assert_frame_equal(final_df, final_df_sorted)
