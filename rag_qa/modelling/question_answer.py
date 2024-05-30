"""Question answering functions"""
from typing import Tuple

import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model_flan(
    model_checkpoint: str = "google/flan-t5-xxl",
    use_gpu: bool = False,
) -> Tuple[
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
    transformers.models.t5.tokenization_t5_fast.T5TokenizerFast,
]:
    """Load in a pre-trained Huggingface model and tokenizer for Question Answering

    Args:
        model_checkpoint (str, optional): reference to the pre-trained Huggingface model to load.
        Defaults to "google/flan-t5-xxl".
        use_gpu (bool, optional): flag to determine whether a GPU is used. Defaults to False.

    Returns:
        Tuple[
            transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
            transformers.models.t5.tokenization_t5_fast.T5TokenizerFast,
            ]: pre-trained Huggingface model and tokenizer
    """
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    if use_gpu:
        model = T5ForConditionalGeneration.from_pretrained(
            model_checkpoint, device_map="auto", torch_dtype=torch.float16
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    return model, tokenizer


def answer_question_flan(
    model: transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
    tokenizer: transformers.models.t5.tokenization_t5_fast.T5TokenizerFast,
    context: str,
    question: str,
    use_gpu: bool = False,
    min_length: int = 200,
    max_length: int = 500,
    no_repeat_ngram_size: int = 7,
) -> str:
    """Generate an answer to a question based on a specified context

    Args:
        model (transformers.models.t5.modeling_t5.T5ForConditionalGeneration): the model used to generate an
        answer to the question
        tokenizer (transformers.models.t5.tokenization_t5_fast.T5TokenizerFast): the tokenizer used to generate
        an answer to the question
        context (str): the context from which the answer should be generated
        question (str): the question to be answered
        use_gpu (bool): Use of GPU or not
        min_length (int): min no of tokens in generated answer
        max_length (int): max no of tokens in generated answer
        no_repeat_ngram_size (int): no n-gram appears twice in generated answer

    Returns:
        str: the answer to the question
    """
    prompt = f"{context}\nAnswer this question: {question} Do not repeat sentences to satisfy the minimum output length."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if use_gpu:
        input_ids = input_ids.to("cuda")

    output = tokenizer.decode(
        model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )[0]
    )

    return output
