"""Preprocessing functions"""
import os
import re

# import shutil
from typing import List, Tuple

import datasets
import faiss
import fitz
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset


def fonts(doc: fitz.fitz.Document) -> Tuple[list, dict]:
    """Extracts fonts and their usage in PDF documents.

    Args:
        doc (fitz.fitz.Document): PDF document to iterate through

    Raises:
        ValueError: no fonts are found

    Returns:
        Tuple[list, dict]:
        font_counts (list): most used fonts sorted by count,
        styles (dict): all text styles found within the document
    """
    styles = {}
    font_counts: dict = {}

    blocks = [block for page in doc for block in page.get_text("dict")["blocks"]]
    lines = [line for block in blocks if block["type"] == 0 for line in block["lines"]]
    spans = [span for line in lines for span in line["spans"]]

    for span in spans:  # iterate through the text spans
        identifier = f'{span["size"]}_{span["font"]}_{span["color"]}'
        styles[identifier] = {
            "size": span["size"],
            "font": span["font"],
            "color": span["color"],
        }
        font_counts[identifier] = (
            font_counts.get(identifier, 0) + 1
        )  # count the fonts usage

    font_counts_list = list(font_counts.items())
    font_counts_list = sorted(font_counts_list, key=lambda x: x[1], reverse=True)

    if len(font_counts_list) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts_list, styles


def font_style_flags(p_colour: float, colour: float, font: str) -> Tuple[str, str]:
    """Creates colour and bold/italic flags for a font

    Args:
        p_colour (float): colour of the document body text
        colour (float): colour of the text
        font (str): the font of the text

    Returns:
        Tuple[str, str]:
        colour_f (str): colour flag
        bold_italic_f (str): bold/italic flag
    """
    bold_italic = re.search("Bold|Italic", font, re.I)
    if colour == p_colour:
        colour_f = "No"
    else:
        colour_f = "Yes"
    if bold_italic:
        bold_italic_f = "Yes"
    else:
        bold_italic_f = "No"
    return colour_f, bold_italic_f


def font_styles(font_counts: list, styles: dict, p_colour: float) -> pd.DataFrame:
    """Creates a pandas dataframe containing each font size, colour flag and bold/italic flag
    for each font style within the document

    Args:
        font_counts (list): most used fonts sorted by count
        styles (dict): all text styles found within the document
        p_colour (float): colour of the document body text

    Returns:
        pd.DataFrame: a pandas dataframe containing each font size, colour flag and bold/italic flag
    """
    font_sizes = []
    colours = []
    bold_italics = []
    for font, _ in font_counts:
        size = float(styles[font]["size"])
        colour = float(styles[font]["color"])
        colour_f, bold_italic_f = font_style_flags(
            p_colour, colour, styles[font]["font"]
        )
        font_sizes.append(size)
        colours.append(colour_f)
        bold_italics.append(bold_italic_f)

    font_style = pd.DataFrame(
        zip(font_sizes, colours, bold_italics),
        columns=["Size", "Flag_Colour", "Flag_Bold_Italic"],
    )
    font_style = font_style.sort_values(
        ["Size", "Flag_Colour", "Flag_Bold_Italic"], ascending=False
    ).reset_index(drop=True)
    font_style = font_style.drop_duplicates().reset_index(drop=True)

    return font_style


def font_size_tag(font_style: pd.DataFrame, p_size: float, doc_type: str) -> dict:
    """Creates a dictionary of font tags for each font style present in the document

    Args:
        font_style (pd.DataFrame): a pandas dataframe containing each font size, colour flag and bold/italic flag
        p_size (float): size of the document body text
        doc_type (str): type of document being processed

    Returns:
        dict: textual element tags for each font size
    """
    idx = 0
    size_tag = {}
    for i in range(font_style.shape[0]):
        size = font_style["Size"][i]
        colour_f = font_style["Flag_Colour"][i]
        bold_italic_f = font_style["Flag_Bold_Italic"][i]
        idx += 1
        size_colour_bold_italic_f = str(size) + "_" + colour_f + "_" + bold_italic_f
        if (
            (size == p_size)
            and (bold_italic_f == "No")
            and (doc_type == "Audit Standard")
        ):
            size_tag[size_colour_bold_italic_f] = "<p>"
        elif (size == p_size) and (colour_f == "No") and (doc_type == "FAQ"):
            size_tag[size_colour_bold_italic_f] = "<p>"
        elif (size == p_size) and (colour_f == "Yes") and (doc_type == "FAQ"):
            size_tag[size_colour_bold_italic_f] = f"<c{idx}>"
        elif (size > p_size) and (colour_f == "Yes") and (doc_type == "KAEG"):
            size_tag[size_colour_bold_italic_f] = f"<ch{idx}>"
        elif (size == p_size) and (colour_f == "Yes") and (doc_type == "KAEG"):
            size_tag[size_colour_bold_italic_f] = f"<c{idx}>"
        elif (size == p_size) and (doc_type == "KAEG"):
            size_tag[size_colour_bold_italic_f] = "<p>"
        elif (size > p_size) or (bold_italic_f == "Yes"):
            size_tag[size_colour_bold_italic_f] = f"<h{idx}>"
        elif size < p_size:
            size_tag[size_colour_bold_italic_f] = f"<s{idx}>"

    return size_tag


def font_tags(
    doc: fitz.fitz.Document, font_counts: list, styles: dict
) -> Tuple[dict, dict]:
    """Creates a dictionary with font sizes as keys and tags as value.

    Args:
        doc (fitz.fitz.Document): PDF document to iterate through
        font_counts (list): most used fonts sorted by count
        styles (dict): all text styles found within the document

    Returns:
        Tuple[dict, dict]:
        size_tag (dict): textual element tags for each font size,
        p_style (dict): textual style of the document body text
    """
    doc_name = os.path.basename(doc.name)
    p_style = styles[
        font_counts[0][0]
    ]  # get style for most used font by count (paragraph)
    p_size = float(p_style["size"])  # get the paragraph's size
    p_colour = float(p_style["color"])
    font_style = font_styles(font_counts, styles, p_colour)
    # aggregating the tags for each font size
    if doc_type_check := re.search("FAQ|Audit Standard|KAEG", doc_name):
        doc_type = doc_type_check[0]
    else:
        doc_type = "Audit Standard"  # Other docs should be treated as ISA for now
    size_tag = font_size_tag(font_style, p_size, doc_type)

    return size_tag, p_style


def headers_para(
    doc: fitz.fitz.Document, size_tag: dict, p_style: dict
) -> List[Tuple[str, int]]:
    """Scrapes headers & paragraphs from PDF and return texts with element tags.

    Args:
        doc (fitz.fitz.Document): PDF document to iterate through
        size_tag (dict): textual element tags for each font size
        p_style (dict): textual style of the document body text

    Returns:
        List[str]: list of texts with prepended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    blocks = [
        (block, page.number + 1)
        for page in doc
        for block in page.get_text("dict")["blocks"]
        if block["type"] == 0
    ]

    for block, page in blocks:  # iterate through the text blocks
        # REMEMBER: multiple fonts and sizes are possible IN one block
        block_string = ""  # text found in block
        spans = [span for line in block["lines"] for span in line["spans"]]
        for span in spans:
            colour_f, bold_italic_f = font_style_flags(
                p_style["color"], span["color"], span["font"]
            )
            tag = str(span["size"]) + "_" + colour_f + "_" + bold_italic_f
            if span["text"].strip() and first:  # removing whitespaces:
                previous_s = span
                first = False
                block_string = size_tag[tag] + span["text"]
            elif span["text"].strip() and not first:
                if span["size"] == previous_s["size"]:
                    if block_string and all((c == "|") for c in block_string):
                        # block_string only contains pipes
                        block_string = size_tag[tag] + span["text"]
                    if block_string == "":
                        # new block has started, so append size tag
                        block_string = size_tag[tag] + span["text"]
                    else:  # in the same block, so concatenate strings
                        block_string += " " + span["text"]
                else:
                    header_para.append((block_string, page))
                    block_string = size_tag[tag] + span["text"]

                previous_s = span

            # new block started, indicating with a pipe
            block_string += "|"

        header_para.append((block_string, page))

    return header_para


def clean_text(text: str) -> str:
    """Removes font tags from text

    Args:
        text (str): text line from a document

    Returns:
        str: clean text line
    """
    text_clean = re.sub(r"<h\d+>|<p>|<c\d+>|<ch\d+>|\|", "", text)
    text_clean = text_clean.strip()
    return text_clean


def clean_doc(text_lines: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filters out unnecessary text lines

    Args:
        text_lines (List[str]): list of text lines

    Returns:
        List[str]: list of text lines
    """
    # remove blank lines
    text_lines = [(text, page) for (text, page) in text_lines if text != ""]
    # remove subscript lines
    text_lines = [
        (text, page) for (text, page) in text_lines if not text.startswith("<s")
    ]
    # remove short lines
    text_lines = [(text, page) for (text, page) in text_lines if len(text) > 10]
    # remove index lines
    text_lines = [
        (text, page)
        for (text, page) in text_lines
        if re.search(r"\.\.\.+", text) is None
    ]
    return text_lines


def combine_lines_recursively(
    text_lines: List[Tuple[str, int]], start_tag: str, end_tag: str
) -> List[Tuple[str, int]]:
    """Combines adjacent text lines with specified font tags recursively

    Args:
        text_lines (List[str]): list of text lines
        start_tag (str): font tag of text line to append to
        end_tag (str): font tag of text line to prepend to

    Returns:
        List[str]: list of text lines
    """
    new_text_lines = []
    i = 0
    while i < len(text_lines):
        current = text_lines[i][0]
        # The combined text lines will take page number of the first line:
        page = text_lines[i][1]
        if current.startswith(start_tag):
            for j in range(i + 1, len(text_lines)):
                next_word = text_lines[j][0]
                if next_word.startswith(end_tag):
                    current = current + "\n" + next_word
                else:
                    i = j - 1
                    break
                if j == len(text_lines) - 1:
                    i = j - 1
        new_text_lines.append((current, page))
        i += 1

    return new_text_lines


def break_into_para_faq(text_lines: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Splits the FAQ document type into meaningful sections of text

    Args:
        text_lines (List[str]): list of text lines

    Returns:
        List[str]: list of text lines
    """
    text_lines = combine_lines_recursively(text_lines, "<c", "<c")
    text_lines = combine_lines_recursively(text_lines, "<c", "<p")
    text_lines = [
        (clean_text(line), page) for (line, page) in text_lines if line.startswith("<c")
    ]

    return text_lines


def break_into_para_audit(text_lines: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Splits the Audit Standard document type into meaningful sections of text

    Args:
        text_lines (List[str]): list of text lines

    Returns:
        List[str]: list of text lines
    """
    text_lines = combine_lines_recursively(text_lines, "<h", "<p")
    text_lines = [
        (clean_text(line), page) for (line, page) in text_lines if line.startswith("<h")
    ]

    return text_lines


def break_into_para_kaeg(text_lines: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Splits the KAEG document type into meaningful sections of text

    Args:
        text_lines (List[str]): list of text lines

    Returns:
        List[str]: list of text lines
    """

    font_tags_list = list({line.split(">")[0] for (line, page) in text_lines})

    text_lines = [
        (re.sub(r"<c\d+>", "<p>", line), page)
        if re.match(r"<c\d+>", line) and "?" not in line
        else (line, page)
        for (line, page) in text_lines
    ]

    for tag in font_tags_list:
        text_lines = combine_lines_recursively(text_lines, tag, tag)

    for tag in font_tags_list:
        text_lines = combine_lines_recursively(text_lines, tag, "<p")

    text_lines = [(clean_text(line), page) for (line, page) in text_lines]

    return text_lines


def break_into_para(
    text_lines: List[Tuple[str, int]], doc: fitz.fitz.Document
) -> List[Tuple[str, int]]:
    """Identifies the document type to determine the pre-processing method

    Args:
        text_lines (List[str]): list of text lines
        doc (fitz.fitz.Document): PDF document to iterate through

    Returns:
        List[str]: _description_
    """
    doc_name = os.path.basename(doc.name)
    if re.search("FAQ", doc_name):
        text_lines = break_into_para_faq(text_lines)
    elif re.search("KAEG", doc_name):
        text_lines = break_into_para_kaeg(text_lines)
    elif re.search("Audit Standard", doc_name):
        text_lines = break_into_para_audit(text_lines)
    else:
        text_lines = break_into_para_audit(
            text_lines
        )  # Other docs should be treated as ISA for now
    return text_lines


def sliding_window(
    text_lines: List[Tuple[str, int]], max_length: int = 512
) -> List[Tuple[str, int]]:
    """_summary_

    Args:
        text_lines (list): _description_
        max_length (int, optional): _description_. Defaults to 512.

    Returns:
        list: _description_
    """
    new_text_lines = []
    stride = int(round(max_length / 4, 0))
    for text_line, page in text_lines:
        words = text_line.split()
        num_words = len(words)
        if num_words <= max_length:
            new_line = " ".join(words)
            new_text_lines.append((new_line, page))
        else:
            n_chunks = ((num_words - max_length) / stride) + 1
            for i in range(round(n_chunks)):
                start = i * stride
                end = i * stride + max_length
                new_words = words[start:end]
                if len(new_words) == max_length:
                    new_line = " ".join(new_words)
                    new_text_lines.append((new_line, page))
            if n_chunks % 1 != 0:
                start = num_words - max_length
                new_words = words[start:]
                new_line = " ".join(new_words)
                new_text_lines.append((new_line, page))
    return new_text_lines


def get_embeddings(
    text_list: list,
    model: transformers.models.mpnet.modeling_mpnet.MPNetModel,
    tokenizer: transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
) -> torch.tensor:
    """Embed a piece list of text lines with a pre-trained Huggingface model

    Args:
        text_list (list): list of text lines to embed
        model (transformers.models.mpnet.modeling_mpnet.MPNetModel): Pre-trained Huggingface model
        tokenizer (transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast): Pre-trained Huggingface tokenizer

    Returns:
        torch.tensor: torch tensor containing the embeddings for each text line
    """
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = dict(encoded_input.items())  # v.to(device)
    model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0]


def add_faiss_index(
    para_df: pd.DataFrame, vector_method: str, nclus: int
) -> datasets.arrow_dataset.Dataset:
    """Add the FAISS index based on given vector method i.e. Cosine, IVF, FLATL2 etc.

    Args:
        para_df (pd.DataFrame): Dataframe having sections of PDF document along with embeddings
        vector_method: Type of vector method selected for indexing, FlatL2, FlatIP, FlatIVF
        nclus: number of cluster for IVF indexing method
    Returns:
        datasets.arrow_dataset.Dataset: Dataset contains text and embeddings along with index for effiecient search
    """
    # dimension
    dim = para_df["embeddings"][0].shape[0]
    # convert into dataset
    content_dataset = Dataset.from_pandas(para_df)

    if vector_method == "Cosine":
        content_dataset.add_faiss_index(
            column="embeddings", custom_index=faiss.IndexFlatIP(dim)
        )
    elif vector_method == "IVF":
        if nclus > content_dataset.shape[0]:
            nclus = content_dataset.shape[0]

        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nclus)  # nclus
        index.train(np.array(content_dataset["embeddings"]).astype(np.float32))
        content_dataset.add_faiss_index(column="embeddings", custom_index=index)
    else:
        content_dataset.add_faiss_index(column="embeddings")
    return content_dataset


def change_filename(filename: str) -> str:
    """change the file name is contain [ or ] as it creates error while converting into vector indices

    Args:
        filename (str): pdf filename to be processed

    Returns:
        str: filename after replacing the [ or ] with _ so that doesnt interfere with FAISS
    """
    if re.search(r"\[|\]", filename):
        new_name = re.sub(r"\[|\]", "_", filename)
    else:
        new_name = filename
    return new_name


def process_doc(
    doc: fitz.fitz.Document,
    output_path: str,
    model: transformers.models.mpnet.modeling_mpnet.MPNetModel,
    tokenizer: transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
    max_length: int,
    vector_method: str,
    nclus: int,
) -> pd.DataFrame:
    """Load in a PDF file and extract the text into clean, meaningful sections of text.
    Extract embeddings and add a FAISS index for each section and save to file.

    Args:
        doc (fitz.fitz.Document):  PDF document to iterate through
        output_path (str): output directory path to save all the files
        model (transformers.models.mpnet.modeling_mpnet.MPNetModel): Pre-trained Huggingface model
        tokenizer (transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast): Pre-trained Huggingface tokenizer
        vector_method: Type of vector method selected for indexing, FlatL2, FlatIP, FlatIVF
        nclus: number of cluster for IVF indexing method

    Returns:
        pd.DataFrame: pandas dataframe containing the clean sections of text and embeddings
    """
    output_dir = os.path.join(output_path, os.path.basename(os.path.dirname(doc.name)))
    font_counts, styles = fonts(doc)
    size_tag, p_style = font_tags(doc, font_counts, styles)
    elements = headers_para(doc, size_tag, p_style)
    # remove blank lines and subscripts
    text_lines = clean_doc(elements)
    text_lines = break_into_para(text_lines, doc)
    # remove sections with less than 20 words
    text_lines = [
        (text, page) for (text, page) in text_lines if len(text.split(" ")) >= 20
    ]
    # split sentences longer than max_length using a sliding window
    text_lines = sliding_window(text_lines, max_length)
    para_df = pd.DataFrame(text_lines, columns=["content", "page"])
    para_df["embeddings"] = para_df.apply(
        lambda x: get_embeddings(x["content"], model, tokenizer)
        .detach()
        .cpu()
        .numpy()[0],
        axis=1,
    )
    # save dataframe
    os.makedirs(output_dir, exist_ok=True)
    name = re.sub(".pdf", "", os.path.basename(doc.name))
    para_df.to_csv(os.path.join(output_dir, name + ".csv"), index=False)
    # add FAISS index
    content_dataset = add_faiss_index(para_df, vector_method, nclus)
    # saving Faiss index
    content_dataset.save_faiss_index(
        "embeddings", os.path.join(output_dir, name + "_my_index.faiss")
    )
    return para_df


def process_docs(
    infiles: List[str],
    output_path: str,
    model: transformers.models.mpnet.modeling_mpnet.MPNetModel,
    tokenizer: transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast,
    max_length: int,
    vector_method: str = "FlatL2",
    nclus: int = 3,
) -> List[pd.DataFrame]:
    """Load in multiple files and pre-process them

    Args:
        infiles (List[str]): list of input filepaths to be processed
        output_path (str): output directory path to save all the files
        model (transformers.models.mpnet.modeling_mpnet.MPNetModel): Pre-trained Huggingface model
        tokenizer (transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast): Pre-trained Huggingface tokenizer
        vector_method: Type of vector method selected for indexing, FlatL2, FlatIP, FlatIVF
        nclus: number of cluster for IVF indexing method

    Returns:
        List[pd.DataFrame]: list of dataframes containing the clean sections of text and embeddings
    """
    para_dfs = []
    # first fix the filename if any has [ or ]
    new_files = []
    for fl in infiles:
        os.rename(fl, change_filename(fl))
        new_files.append(change_filename(fl))
    # preprocess the files
    for file in new_files:
        print("Processing file:", file)
        doc = fitz.open(file)
        para_df = process_doc(
            doc, output_path, model, tokenizer, max_length, vector_method, nclus
        )
        para_dfs.append(para_df)
    return para_dfs
