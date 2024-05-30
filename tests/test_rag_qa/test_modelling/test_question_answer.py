# pylint: disable=missing-module-docstring, missing-function-docstring
from unittest.mock import Mock, patch  # MagicMock

import torch

from rag_qa.modelling.question_answer import (
    answer_question_flan,
    load_model_flan,
)

def test_answer_question_flan(mocker: Mock) -> None:
    context = "This is the context."
    question = "What is the answer?"
    use_gpu = True
    min_length = 200
    max_length = 500
    no_repeat_ngram_size = 7

    # Mock tokenizer.__call__ (tokenizer.encode)
    input_ids_mock = mocker.Mock()
    tokenizer = mocker.Mock(return_value=input_ids_mock)

    # Mock model.generate
    generated_ids_mock = mocker.Mock()
    model = mocker.Mock()
    model.generate.return_value = [generated_ids_mock]

    # Mock tokenizer.decode
    decoded_output_mock = mocker.Mock()
    tokenizer.decode.return_value = decoded_output_mock

    answer = answer_question_flan(
        model,
        tokenizer,
        context,
        question,
        use_gpu,
        min_length,
        max_length,
        no_repeat_ngram_size,
    )

    # Assertions
    assert answer == decoded_output_mock

    tokenizer.assert_called_once_with(
        f"{context}\nAnswer this question: {question} Do not repeat sentences to satisfy the minimum output length.",
        return_tensors="pt",
    )

    model.generate.assert_called_once_with(
        input_ids_mock.input_ids.to(),
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    tokenizer.decode.assert_called_once_with(generated_ids_mock)
