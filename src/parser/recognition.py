import os

from .base import add_cuda_options, add_misc_options, adding_cuda, argparse
from .checkpoint import construct_checkpointname
from .dataset import add_dataset_options
from .tools import save_args
from .training import add_training_options


def training_parser():
    parser = argparse.ArgumentParser()

    # misc options
    add_misc_options(parser)

    # training options
    add_training_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_cuda_options(parser)

    opt = parser.parse_args()

    # remove None params, and create a dictionary
    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    parameters["modelname"] = "recognition"

    if "folder" not in parameters:
        parameters["folder"] = construct_checkpointname(parameters, parameters["expname"])

    os.makedirs(parameters["folder"], exist_ok=True)
    save_args(parameters, folder=parameters["folder"])

    adding_cuda(parameters)

    return parameters
