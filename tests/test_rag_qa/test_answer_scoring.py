# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest.mock import Mock, patch

import pandas as pd

from dpp_helpline_qa.answer_scoring import arg_parser, main
from dpp_helpline_qa.modelling.semantic_search import load_model_ss


def test_arg_parser() -> None:
    test_args = ["--filepath", "test_filepath.csv"]

    args = arg_parser(test_args)

    assert args.filepath == test_args[1]


@patch("pandas.read_csv")
@patch("dpp_helpline_qa.answer_scoring.answer_scoring")
@patch("dpp_helpline_qa.answer_scoring.load_model_ss")
def test_main(
    load_model_ss_mock: Mock, answer_scoring_mock: Mock, read_csv_mock: Mock
) -> None:
    test_df = pd.DataFrame(
        {
            "Answer": ["This is the actual answer"],
            "Generated Answer": ["This is the generated answer"],
            "Extracted Context": ["This is the extracted context"],
        }
    )
    read_csv_mock.return_value = test_df
    load_model_ss_mock.return_value = load_model_ss()

    with patch("sys.argv", ["", "--filepath", "test_filepath.csv"]):
        main()

    read_csv_mock.assert_called_once_with("test_filepath.csv")
    load_model_ss_mock.assert_called_once()
    answer_scoring_mock.assert_called_once()
