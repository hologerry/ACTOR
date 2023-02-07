from src.datasets.get_dataset import get_datasets
from src.models.get_model import get_model as get_gen_model
from src.recognition.get_model import get_model as get_rec_model


def get_model_and_data(parameters):
    datasets = get_datasets(parameters)

    if parameters["modelname"] == "recognition":
        model = get_rec_model(parameters)
    else:
        model = get_gen_model(parameters)
    return model, datasets
