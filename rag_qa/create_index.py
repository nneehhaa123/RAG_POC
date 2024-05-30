"""Create index for files in the specified input directory and store the results in the specified output directory"""
import argparse
import glob
import os
import re
import shutil
from typing import List, Optional

import yaml

from rag_qa.modelling.semantic_search import load_model_ss
from rag_qa.preprocessing.preprocessing import process_docs


def unindexed(infiles: List[str], indir: str, outdir: str) -> List[str]:
    """When an incremental run has been specified, obtain the list of files which have not yet been indexed.

    Args:
        infiles (List[str]): list of files in the input directory
        indir (str): path of the input directory
        outdir (str): path of the output directory

    Returns:
        list(str): list of as yet unindexed files
    """
    # Strip any trailing "/" from path
    indir = re.sub("/$", "", indir)
    outdir = re.sub("/$", "", outdir)

    # Strip .pdf extensions and start of path
    in_set = {re.sub(".pdf$", "", file) for file in infiles}
    in_set = {re.sub("^" + indir, "", file) for file in in_set}

    indexed_files = glob.glob(os.path.join(outdir, "**"), recursive=True)
    indexed_files = [f for f in indexed_files if os.path.isfile(f)]

    # Strip out any file extension and start of path
    out_set = {re.sub(".faiss|.csv", "", file) for file in indexed_files}
    out_set = {re.sub("^" + outdir, "", file) for file in out_set}

    diff = in_set - out_set

    # Add input directory and .pdf extension
    files = [indir + file + ".pdf" for file in diff]

    return files


def get_infile_names(indir: str, outdir: str, incremental: bool = False) -> List[str]:
    """
    Get list of files that need to be indexed.

    Args:
        indir (str): Input directory
        outdir (str): Output directory
        incremental (bool): If True, only index new files from the input directory.
            If False, delete output directory and process all the files in the input directory.

    Returns:
        List[str]: list of files to be indexed
    """
    files = glob.glob(os.path.join(indir, "**", "*.pdf"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if incremental:
        files = unindexed(files, indir, outdir)
    else:
        # Delete folder
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)

    return files


def arg_parser(args: Optional[list] = None) -> argparse.Namespace:
    """Parse the command line arguments

    Args:
        args (list, optional): list of command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: the parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="RAG QA tool")
    parser.add_argument("--indir", help="Directory containing input files")
    parser.add_argument(
        "--outdir",
        default="output",
        help="Directory to place tokenized and indexed data",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="If set, only index files which have not already been indexed",
    )
    return parser.parse_args(args)


def main() -> None:
    """
    Deal with commandline options, then call create_index.

    Returns:
        None
    """

    args = arg_parser()

    if args.indir:
        infiles = get_infile_names(
            args.indir, args.outdir, incremental=args.incremental
        )

    create_index(infiles, args.outdir)


def create_index(infiles: List[str], outdir: str) -> None:
    """
    Create index for the files listed in infiles.

    Args:
        infiles (List[str]): files from the input directory to be indexed
        outdir (str): output directory

    Returns:
        None
    """
    with open("config/config.yaml", encoding="utf-8") as cf_file:
        config = yaml.safe_load(cf_file.read())

    max_length = config["preprocessing"]["max_context_length"]
    vector_method = config["preprocessing"]["vector_simalirity_method"]
    nclus_ivf = config["preprocessing"]["nclus_ivf"]
    model_semantic = config["modelling"]["semantic_model"]

    # load the model and tokenizer for semantic search
    model_ss, tokenizer_ss = load_model_ss(model_semantic)

    # load and pre-process the documents to prepare for searching
    _ = process_docs(
        infiles, outdir, model_ss, tokenizer_ss, max_length, vector_method, nclus_ivf
    )


if __name__ == "__main__":
    main()
