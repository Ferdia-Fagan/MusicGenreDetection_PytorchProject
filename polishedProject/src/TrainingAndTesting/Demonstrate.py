import json
import os
import torch

from .. import *

from src.TrainingAndTesting import Utils, TrainAndTest, MusicDataLoader
from src.utils import HParams, Plotting, Main

root_directory = f"{os.getcwd()}"

LOCATION_TO_MODEL_PARAMS="model_params"

model_name = "CNN"

root_directory = f"{os.path.dirname(__file__)}/.."

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def loadModelWithState(modelToInstantiate, modelStateAddress, params):

    if os.path.exists(modelStateAddress):
        # print("loading previous state")
        # torch.save(model.state_dict(),
        #            f"{root_directory}/ModelStates/{params.model_state_save_directory}/{params.model_name}")
        model = modelToInstantiate.Net(params)
        model.load_state_dict(torch.load(modelStateAddress))
        # model.to(device)
        model.eval()
        return model
    else:
        return None


def get_Params_And_DasetPath(model_name_testing, modelDirectory):
    model_parameters_testing = HParams.loadModelParameters(modelDirectory,model_name_testing)

    datasetPath = f"{root_directory}/../../{model_parameters_testing.data_dir}"

    return [model_parameters_testing, datasetPath]



def demonstrateModel(modelName, modelToDemonstrate, modelDirectory, redemonstrateWithLog=True ):
    print("testing model: " + modelName)
    modelWeights_fileLoc = Utils.getParamsForModel(modelDirectory, modelName)

    model_parameters_testing, datasetPath = get_Params_And_DasetPath(modelName, modelDirectory)

    torch.cuda.empty_cache()
    _, _, test_dataloader = MusicDataLoader.get_TrainAndTestDatasets(datasetPath, 100, 0.3, 0.2)


    # show loss and accuracy graphs
    modelLogPath = Utils.getLogsForModel(modelDirectory, modelName)

    with open(modelLogPath) as json_file:
        modelLogs = json.load(json_file)

    model = loadModelWithState(modelToDemonstrate, modelWeights_fileLoc, model_parameters_testing)

    Main.runModelWithSampleDataToGetTensorBoardFile(model,modelName,test_dataloader)


    # model.to(device)

    # test
    if not redemonstrateWithLog:

        demoAccuracy = TrainAndTest.test(model, device, test_dataloader)
        print("demo accuracy: ", demoAccuracy)
    else:
        print("The final training accuracy was: ", modelLogs['train_accs'][-1])
        print("The final validation accuracy was: ", modelLogs['val_accs'][-1])
        # bestEpoc_i = modelLogs["best_val_epoch"]
        # test_acc = modelLogs["test_acc"]
        # print("demo accuracy: ", test_acc)

    Plotting.plot_training(modelLogs['train_losses'], modelLogs['train_accs'],
                           modelLogs['val_losses'], modelLogs['val_accs'],
                           modelName)
















