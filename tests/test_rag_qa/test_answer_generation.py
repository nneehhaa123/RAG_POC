# pylint: disable=missing-module-docstring, missing-function-docstring
import os
from unittest.mock import Mock, patch  # MagicMock

import numpy as np
import pandas as pd

from dpp_helpline_qa.answer_generation import arg_parser, generate_answers, main


def test_arg_parser() -> None:
    test_args = ["--filepath", "test_filepath.csv", "--index_dir", "output"]

    args = arg_parser(test_args)

    assert args.filepath == test_args[1]
    assert args.index_dir == test_args[3]


@patch("dpp_helpline_qa.modelling.semantic_search.AutoTokenizer")
@patch("dpp_helpline_qa.modelling.semantic_search.AutoModel")
@patch("dpp_helpline_qa.modelling.question_answer.T5Tokenizer")
@patch("dpp_helpline_qa.modelling.question_answer.T5ForConditionalGeneration")
@patch("dpp_helpline_qa.answer_generation.context_ranking")
def test_generate_answers(
    tokenizer_ss: Mock,
    model_ss: Mock,
    tokenizer_qa: Mock,
    model_qa: Mock,
    context_ranking_mock: Mock,
) -> None:
    ques_file = os.path.join("tests", "test_data", "test_df.csv")
    df = pd.read_csv(ques_file)

    context_df = pd.DataFrame(
        {
            "content": ["This is sample1", "This is sample2"],
            "page": [1, 2],
            "embeddings": [np.random.rand(768), np.random.rand(768)],
            "scores": [0.67, 0.78],
            "doc_name": ["sample_FAQ.pdf", "sample_KAEG.pdf"],
            "temp_cont": ["this is sample1", "this is sample2"],
        }
    )

    context_ranking_mock.return_value = context_df
    out = generate_answers(df, "output", model_ss, tokenizer_ss, model_qa, tokenizer_qa)

    assert list(out.columns) == [
        "Question",
        "Extracted Context",
        "Generated Answer",
        "ContextRef_1",
        "ContextRef_2",
        "ContextRef_3",
        "ContextRef_4",
        "ContextRef_5",
    ]
    assert out.shape[0] == context_df.shape[0]


@patch("pandas.read_csv")
@patch("dpp_helpline_qa.answer_generation.load_model_ss")
@patch("dpp_helpline_qa.answer_generation.load_model_flan")
@patch("dpp_helpline_qa.answer_generation.generate_answers")
def test_main(
    read_csv_mock: Mock,
    load_model_ss_mock: Mock,
    load_model_flan_mock: Mock,
    generate_answers_mock: Mock,
) -> None:
    test_df = pd.DataFrame({"Question": ["This is the question"]})
    read_csv_mock.return_value = test_df
    # Mock tokenizer.__call__ (tokenizer.encode)
    input_ids_mock0 = Mock()
    tokenizer_ss = Mock(return_value=input_ids_mock0)

    # Mock model.generate
    generated_ids_mock0 = Mock()
    model_ss = Mock()
    model_ss.last_hidden_state.return_value = [generated_ids_mock0]

    # Mock tokenizer.__call__ (tokenizer.encode)
    input_ids_mock = Mock()
    tokenizer_qa = Mock(return_value=input_ids_mock)

    # Mock model.generate
    generated_ids_mock = Mock()
    model_qa = Mock()
    model_qa.generate.return_value = [generated_ids_mock]

    # Mock tokenizer.decode
    decoded_output_mock = Mock()
    tokenizer_qa.decode.return_value = decoded_output_mock

    load_model_ss_mock.return_value = (model_ss, tokenizer_ss)
    load_model_flan_mock.return_value = (model_qa, tokenizer_qa)

    with patch(
        "sys.argv", ["", "--filepath", "test_filepath.csv", "--index_dir", "output"]
    ):
        main()

    load_model_ss_mock.assert_called_once()
    load_model_flan_mock.assert_called_once()
    generate_answers_mock.assert_called_once()
