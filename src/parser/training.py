import os

from src.parser.base import (
    ArgumentParser,
    add_cuda_options,
    add_misc_options,
    adding_cuda,
)
from src.parser.checkpoint import construct_checkpointname
from src.parser.dataset import add_dataset_options
from src.parser.model import add_model_options, parse_modelname
from src.parser.tools import save_args


def add_training_options(parser):
    group = parser.add_argument_group("Training options")
    group.add_argument("--batch_size", type=int, required=True, help="size of the batches")
    group.add_argument("--num_epochs", type=int, required=True, help="number of epochs of training")
    group.add_argument("--lr", type=float, required=True, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, required=True, help="frequency of saving model/viz")


def parser():
    parser = ArgumentParser()

    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)

    # training options
    add_training_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    opt = parser.parse_args()

    # remove None params, and create a dictionnary
    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    # parse modelname
    ret = parse_modelname(parameters["modelname"])
    parameters["modeltype"], parameters["archiname"], parameters["losses"] = ret

    # update lambdas params
    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    parameters["lambdas"] = lambdas

    if "folder" not in parameters:
        parameters["folder"] = construct_checkpointname(parameters, parameters["expname"])

    os.makedirs(parameters["folder"], exist_ok=True)
    save_args(parameters, folder=parameters["folder"])

    adding_cuda(parameters)

    return parameters
