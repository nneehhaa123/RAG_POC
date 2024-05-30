# pylint: disable=missing-module-docstring, missing-function-docstring
import glob
import os
from unittest.mock import Mock, patch

from rag_qa.create_index import arg_parser, main, unindexed
from rag_qa.modelling.semantic_search import load_model_ss


def test_arg_parser() -> None:
    test_args = ["--indir", "test_data", "--outdir", "output"]
    args = arg_parser(test_args)
    assert args.indir == test_args[1]
    assert args.outdir == test_args[3]


@patch("rag_qa.create_index.glob")
def test_unindexed(glob_mock: Mock) -> None:
    indir = "test_data"
    outdir = "ouput"
    infiles = [
        "Documentation\\Sample_Audit Standard.pdf",
        "Documentation\\sample_FAQs.pdf",
        "Documentation\\sample_KAEG.pdf",
    ]

    out_files = ["Documentation\\sample_FAQs.csv", "Documentation\\sample_KAEG.faiss"]

    glob_mock.glob.return_value = out_files
    infiles = [f for f in infiles if os.path.isfile(f)]
    output = unindexed(infiles, indir, outdir)

    assert len(output) == 0


@patch("dpp_helpline_qa.create_index.load_model_ss")
def test_main(load_model_ss_mock: Mock) -> None:
    indir_path = os.path.join("tests", "test_data")
    outdir_path = os.path.join("tests", "output")
    load_model_ss_mock.return_value = load_model_ss()

    with patch("sys.argv", ["", "--indir", indir_path, "--outdir", outdir_path]):
        main()
    in_file = glob.glob(os.path.join(indir_path, "*.pdf"))
    out_file_csv = glob.glob(os.path.join(outdir_path, "test_data", "*.csv"))
    out_file_faiss = glob.glob(os.path.join(outdir_path, "test_data", "*.faiss"))

    assert len(in_file) == len(out_file_csv)
    assert len(in_file) == len(out_file_faiss)
