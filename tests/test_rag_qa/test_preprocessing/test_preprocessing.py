# pylint: disable=missing-module-docstring, missing-function-docstring
import os
from unittest.mock import Mock, patch

import datasets
import fitz
import numpy as np
import pandas as pd

from rag_qa.preprocessing.preprocessing import (
    add_faiss_index,
    break_into_para,
    break_into_para_audit,
    break_into_para_faq,
    break_into_para_kaeg,
    change_filename,
    clean_doc,
    clean_text,
    combine_lines_recursively,
    font_size_tag,
    font_style_flags,
    font_styles,
    font_tags,
    fonts,
    get_embeddings,
    process_docs,
    sliding_window,
)


def test_font_style_flags() -> None:
    p_colour = 0
    colours = [0, 1, 0, 1]
    fonts_list = ["regular", "regular", "bold", "Italic"]
    expected_outputs = [("No", "No"), ("Yes", "No"), ("No", "Yes"), ("Yes", "Yes")]

    for colour, font, expected_output in zip(colours, fonts_list, expected_outputs):
        output = font_style_flags(p_colour, colour, font)
        assert output == expected_output


def test_font_styles() -> None:
    font_counts = [
        ("8.625007629394531_ArialMT_4473924", 156),
        ("7.0_TimesNewRomanPSMT_0", 16),
        ("14.55470085144043_ArialMT_539254", 4),
        ("8.625007629394531_Arial-BoldMT_4473924", 1),
    ]
    styles = {
        "8.625007629394531_Arial-BoldMT_4473924": {
            "size": 8.625007629394531,
            "font": "Arial-BoldMT",
            "color": 4473924,
        },
        "8.625007629394531_ArialMT_4473924": {
            "size": 8.625007629394531,
            "font": "ArialMT",
            "color": 4473924,
        },
        "14.55470085144043_ArialMT_539254": {
            "size": 14.55470085144043,
            "font": "ArialMT",
            "color": 539254,
        },
        "7.0_TimesNewRomanPSMT_0": {
            "size": 7.0,
            "font": "TimesNewRomanPSMT",
            "color": 0,
        },
    }
    p_colour = 4473924.0

    expected_output = pd.DataFrame(
        {
            "Size": [14.554701, 8.625008, 8.625008, 7.000000],
            "Flag_Colour": ["Yes", "No", "No", "Yes"],
            "Flag_Bold_Italic": ["No", "Yes", "No", "No"],
        }
    )

    output = font_styles(font_counts, styles, p_colour)

    pd.testing.assert_frame_equal(output, expected_output)


def test_font_tags() -> None:
    file = os.path.join("tests", "test_data", "Documentation", "sample_FAQs" + ".pdf")
    doc = fitz.open(file)
    font_counts, styles = fonts(doc)
    expected_output = (
        {
            "24.25783348083496_Yes_No": "<h1>",
            "14.55470085144043_Yes_No": "<h2>",
            "8.625007629394531_Yes_No": "<c3>",
            "8.625007629394531_No_Yes": "<p>",
            "8.625007629394531_No_No": "<p>",
            "7.0_Yes_No": "<s6>",
            "6.468755722045898_No_No": "<s7>",
        },
        {"size": 8.625007629394531, "font": "ArialMT", "color": 4473924},
    )

    output = font_tags(doc, font_counts, styles)
    assert output == expected_output


def test_font_size_tag() -> None:
    doc_type = "FAQ"
    p_size = 8.625007629394531
    font_style = pd.DataFrame(
        {
            "Size": [24.257833, 8.625008, 8.625008, 8.625008],
            "Flag_Colour": ["Yes", "Yes", "No", "No"],
            "Flag_Bold_Italic": ["No", "No", "Yes", "No"],
        }
    )

    expected_output = {
        "24.257833_Yes_No": "<h1>",
        "8.625008_Yes_No": "<h2>",
        "8.625008_No_Yes": "<h3>",
        "8.625008_No_No": "<h4>",
    }

    output = font_size_tag(font_style, p_size, doc_type)
    assert output == expected_output


def test_clean_text() -> None:
    input_text_1 = "<h5>Hello, world!"
    expected_output_1 = "Hello, world!"

    input_text_2 = "<ch12>This is a test"
    expected_output_2 = "This is a test"

    input_text_3 = "<p>This is another test"
    expected_output_3 = "This is another test"

    input_text_4 = "Not another one|"
    expected_output_4 = "Not another one"

    assert clean_text(input_text_1) == expected_output_1
    assert clean_text(input_text_2) == expected_output_2
    assert clean_text(input_text_3) == expected_output_3
    assert clean_text(input_text_4) == expected_output_4


def test_clean_doc() -> None:
    input_text_lines = [
        ("", 1),
        ("<s>Subscript</s>", 1),
        ("Short line", 1),
        ("This is a line", 1),
        ("This is another line", 2),
        ("Ignore this line...", 2),
    ]

    expected_output = [("This is a line", 1), ("This is another line", 2)]

    output = clean_doc(input_text_lines)

    assert output == expected_output


def test_combine_lines_recursively() -> None:
    text_lines = [
        ("<h1>First sentence", 1),
        ("<h1>Second sentence", 1),
        ("<p>Third sentence", 1),
        ("<p>Fourth sentence", 1),
        ("<h2>Fifth sentence", 2),
        ("<p>Sixth sentence", 2),
        ("<p>Seventh sentence", 2),
        ("<p>Eighth sentence", 2),
        ("<h3>Ninth sentence", 2),
        ("<h4>Tenth sentence", 2),
        ("<p>Eleventh sentence", 2),
        ("<h5>Twelfth sentence", 2),
    ]

    expected_output_1 = [
        ("<h1>First sentence\n<h1>Second sentence", 1),
        ("<p>Third sentence", 1),
        ("<p>Fourth sentence", 1),
        ("<h2>Fifth sentence", 2),
        ("<p>Sixth sentence", 2),
        ("<p>Seventh sentence", 2),
        ("<p>Eighth sentence", 2),
        ("<h3>Ninth sentence\n<h4>Tenth sentence", 2),
        ("<p>Eleventh sentence", 2),
        ("<h5>Twelfth sentence", 2),
    ]

    expected_output_2 = [
        (
            "<h1>First sentence\n<h1>Second sentence\n<p>Third sentence\n<p>Fourth sentence",
            1,
        ),
        (
            "<h2>Fifth sentence\n<p>Sixth sentence\n<p>Seventh sentence\n<p>Eighth sentence",
            2,
        ),
        ("<h3>Ninth sentence\n<h4>Tenth sentence\n<p>Eleventh sentence", 2),
        ("<h5>Twelfth sentence", 2),
    ]

    output_1 = combine_lines_recursively(text_lines, "<h", "<h")
    output_2 = combine_lines_recursively(output_1, "<h", "<p")

    assert output_1 == expected_output_1
    assert output_2 == expected_output_2

@patch("rag_qa.preprocessing.preprocessing.clean_text")
@patch("rag_qa.preprocessing.preprocessing.combine_lines_recursively")
def test_break_into_para_kaeg(
    combine_lines_recursively_mock: Mock, clean_text_mock: Mock
) -> None:
    test_strings = [("<c>This is a test", 1), ("<p>So is this", 1)]

    combine_lines_recursively_mock.return_value = test_strings

    output = break_into_para_kaeg(test_strings)

    assert combine_lines_recursively_mock.call_count == 2 * len(test_strings)

    assert clean_text_mock.call_count == len(output)

def test_sliding_window() -> None:
    text_lines = [
        (
            """<h1>First sentence; <h1>Second sentence, <h1>Second sentence, <h1>Second sentence""",
            1,
        ),
        ("<h1>Second sentence", 1),
    ]
    expected_output = [
        ("<h1>First sentence; <h1>Second sentence, <h1>Second", 1),
        ("sentence; <h1>Second sentence, <h1>Second sentence,", 1),
        ("<h1>Second sentence, <h1>Second sentence, <h1>Second", 1),
        ("sentence, <h1>Second sentence, <h1>Second sentence", 1),
        ("<h1>Second sentence", 1),
    ]
    output = sliding_window(text_lines, 5)
    assert output == expected_output


def test_add_faiss_index() -> None:
    para_df = pd.DataFrame(
        {"embeddings": [np.random.rand(768), np.random.rand(768), np.random.rand(768)]}
    )
    output_1 = add_faiss_index(para_df, vector_method="FlatL2", nclus=3)

    output_2 = add_faiss_index(para_df, vector_method="IVF", nclus=3)

    assert output_1.shape[0] == 3
    assert isinstance(output_1, datasets.arrow_dataset.Dataset)
    assert output_2.shape[0] == 3
    assert isinstance(output_2, datasets.arrow_dataset.Dataset)


def test_change_filename() -> None:
    input_name = "sample[FAQ]123.pdf"
    expected_output = "sample_FAQ_123.pdf"
    output = change_filename(input_name)

    assert output == expected_output


@patch("rag_qa.modelling.semantic_search.AutoTokenizer")
@patch("rag_qa.modelling.semantic_search.AutoModel")
@patch("rag_qa.preprocessing.preprocessing.get_embeddings")
@patch("rag_qa.preprocessing.preprocessing.add_faiss_index")
def test_process_docs(
    tokenizer: Mock, model: Mock, get_embeddings_mock: Mock, add_faiss_index_mock: Mock
) -> None:
    files = [
        os.path.join("tests", "test_data", "Documentation", "sample_FAQs" + ".pdf"),
        os.path.join(
            "tests", "test_data", "Documentation", "Sample_Audit Standard" + ".pdf"
        ),
        os.path.join("tests", "test_data", "Documentation", "sample_KAEG" + ".pdf"),
    ]

    output_path = os.path.join("tests", "output")
    get_embeddings_mock.return_value = np.random.rand(768)
    add_faiss_index_mock.return_value = datasets.arrow_dataset.Dataset

    df = process_docs(
        files,
        output_path,
        model,
        tokenizer,
        max_length=400,
        vector_method="FlatL2",
        nclus=3,
    )

    assert len(df) == 3
    assert list(df[0].columns) == ["content", "page", "embeddings"]
