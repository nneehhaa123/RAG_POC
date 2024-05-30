# pylint: disable=missing-module-docstring, missing-function-docstring
from typing import Tuple

import pandas as pd
import pytest
from nltk.corpus import wordnet

from dpp_helpline_qa.model_validation.model_validation import (
    answer_scoring,
    cal_em_score,
    calculate_semantic_similarity,
    nltk2wn_tag,
    text_preprocess,
)
from dpp_helpline_qa.modelling.semantic_search import load_model_ss


def test_nltk2wn_tag() -> None:
    tags = ["J", "V", "N", "R", "Z", ""]

    assert nltk2wn_tag(tags[0]) == wordnet.ADJ
    assert nltk2wn_tag(tags[1]) == wordnet.VERB
    assert nltk2wn_tag(tags[2]) == wordnet.NOUN
    assert nltk2wn_tag(tags[3]) == wordnet.ADV
    assert nltk2wn_tag(tags[4]) is None
    assert nltk2wn_tag(tags[5]) is None


def test_text_preprocess() -> None:
    string = "The quickest brown foxes jumped over the lazier dogs"

    expected_output = [
        "The",
        "quick",
        "brown",
        "fox",
        "jump",
        "over",
        "the",
        "lazy",
        "dog",
    ]

    output = text_preprocess(string)

    assert output == expected_output


def test_cal_em_score() -> None:
    string_1 = "This is a test sentence"
    string_2 = "This is also a test sentence"

    expected_output = 0.83

    output = cal_em_score(string_1, string_2)

    assert output == expected_output


@pytest.fixture(name="ss_model_tokenizer")
def semantic_search_fixture() -> Tuple:
    model, tokenizer = load_model_ss()
    return model, tokenizer


def test_calculate_semantic_similarity(ss_model_tokenizer: Tuple) -> None:
    model, tokenizer = ss_model_tokenizer

    string_1 = "This is a test sentence"
    string_2 = "This is also a test sentence"

    expected_output = [0.98, 0.97, 0.99]

    output = calculate_semantic_similarity(model, tokenizer, string_1, string_2)

    assert output == expected_output


def test_answer_scoring(ss_model_tokenizer: Tuple) -> None:
    answers = pd.DataFrame(
        {
            "Answer": ["Answer 1", "Answer 2"],
            "Generated Answer": ["Generated Answer 1", "Generated Answer 2"],
            "Extracted Context": ["Extracted Context 1", "Extracted Context 2"],
        }
    )

    expected_scores = {
        "EM_Score_ans": [1.0, 1.0],
        "Sbert_score_ans": [0.48, 0.46],
        "NLP_score_ans": [0.95, 0.95],
        "EM_Score_context": [0.5, 0.5],
        "Sbert_score_context": [0.13, 0.11],
        "NLP_score_context": [0.89, 0.88],
    }

    model, tokenizer = ss_model_tokenizer

    answers = answer_scoring(answers, model, tokenizer)

    assert isinstance(answers, pd.DataFrame)

    assert len(answers.columns) == 9

    for col, score in expected_scores.items():
        assert answers[col].tolist() == score
