import json
import os

import numpy as np
import torch
import torch.optim as optim

from src.utils.HParams import HParams
from src.utils import Plotting

from src.CNN_1D import RawAudio_CNN

from src.CNN_2D import MelSpectogram_CNN as MelSpectogram_2DCNN

from src.LSTM import BasicLSTM
from src.LSTM import RCNN_1D
from src.LSTM import RCNN_2D

from src.TrainingAndTesting import TrainAndTest
from src.TrainingAndTesting import MusicDataLoader

from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

"""

I TOOK this from SAMPLE PROJECT MAIN AND AJUSTED IT.

"""


def setUpReproducibility():
    torch.manual_seed(0)
    np.random.seed(0)


LOCATION_TO_MODEL_PARAMS="model_params"

model_name = "CNN"

root_directory = f"{os.path.dirname(__file__)}/.."

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


def runModelWithSampleDataToGetTensorBoardFile(model, modelName, sampleData_loader):
    writer = SummaryWriter(f"./tensorboardLogs/{modelName}/")

    # train_loader, val_loader, test_loader = \
    #     MusicDataLoader.get_TrainAndTestDatasets(datasetPath,
    #                                                     params.batch_size,
    #                                                     params.testSize,
    #                                                     params.validationSize)

    # if os.path.exists(f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}.pt"):
    #     print("loading previous state")
    #     # torch.save(model.state_dict(),
    #     #            f"{root_directory}/ModelStates/{params.model_state_save_directory}/{params.model_name}")
    #     model = modelToInstantiate.Net(params)
    #     model.load_state_dict(torch.load(f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}.pt"))
    #     model.eval()
    # else:
    #     model = modelToInstantiate.Net(params)
    #     model.eval()

    samples, target = next(iter(sampleData_loader))

    writer.add_graph(model, samples)

    writer.close()

# getModelFromModelName = {
#     "CNN_1D_RawMelSpectogram": MelSpectogram_1DCNN,
#     "CNN_1D_RawAudio": RawAudio_CNN,
#
#     "CNN_2D": MelSpectogram_2DCNN,
#
#     "BASIC_LSTM_BestAugmentation": BasicLSTM,
#     "RCNN_1D_BestAugmentation": RCNN_1D,
#     "RCNN_2D_BestAugmentation": RCNN_2D
# }


def runModelWithParams(modelToInstantiate, params, modelName, modelParentDirectory, datasetPath):
    # model = MelSpectogram_2DCNN.Net(params)

    if not os.path.exists(f"{modelParentDirectory}/ModelStates"):
        os.makedirs(f"{modelParentDirectory}/ModelStates")
    if not os.path.exists(f"{modelParentDirectory}/logs"):
        os.makedirs(f"{modelParentDirectory}/logs")
    if not os.path.exists(f"{modelParentDirectory}/figs"):
        os.makedirs(f"{modelParentDirectory}/figs")

    if os.path.exists(f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}.pt"):
        print("loading previous state")
        # torch.save(model.state_dict(),
        #            f"{root_directory}/ModelStates/{params.model_state_save_directory}/{params.model_name}")
        model = modelToInstantiate.Net(params)
        model.load_state_dict(torch.load(f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}.pt"))
        model.eval()
    else:
        model = modelToInstantiate.Net(params)


    if not os.path.exists(f"{modelParentDirectory}/ModelStates/{modelName}"):
        os.makedirs(f"{modelParentDirectory}/ModelStates/{modelName}")


    # model = RawAudio_CNN.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    train_loader, val_loader, test_loader = \
        MusicDataLoader.get_TrainAndTestDatasets(datasetPath,
                                                        params.batch_size,
                                                        params.testSize,
                                                        params.validationSize)

    model, val_accs, val_losses, train_losses, train_accs = runTrainingAndTesting(model,optimizer,train_loader, val_loader,
                          params.num_epochs,
                          f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}")

    test = TrainAndTest.test

    torch.save(model.state_dict(), f"{modelParentDirectory}/ModelStates/{modelName}/{modelName}.pt")

    if not os.path.exists(f"{modelParentDirectory}/figs/{modelName}"):
        os.makedirs(f"{modelParentDirectory}/figs/{modelName}")
    fig = Plotting.plot_training(train_losses, train_accs, val_losses, val_accs)
    fig.savefig(os.path.join(f"{modelParentDirectory}/figs/{modelName}", "{}_training_vis".format(modelName)))
    test_acc = test(model, device, test_loader)
    # print("The training set loss: ", val_loss)
    print(f"{modelName}: The training set accuracy: ", test_acc)

    logs = {
        "models": modelName,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_epoch": int(np.argmax(val_accs) + 1),
        "test_acc": test_acc,
        # "parameters": params,
        "batch_size": params.batch_size
    }

    with open(os.path.join(f"{modelParentDirectory}/logs", "{}.json".format(modelName)), 'w') as f:
        f.write(json.dumps(logs, indent=4).replace("},", "},\n"))



def runTrainingAndTesting(model,optimizer,
                          train_loader, val_loader,
                          num_epochs,epochStateSave = None):
    train = TrainAndTest.train
    val = TrainAndTest.val


    # train_loader, val_loader, test_loader = \
    #     MusicDataLoader.get_TrainAndTestDatasets(dataset_path,
    #                                                     params.batch_size,
    #                                                     params.testSize,
    #                                                     params.validationSize)

    val_accs = []
    val_losses = []
    train_losses = []
    train_accs = []
    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}".format(epoch))
        # Call training function.
        train(model, device, train_loader, optimizer)
        # Evaluate on both the training and validation set.
        train_loss, train_acc = val(model, device, train_loader)
        val_loss, val_acc = val(model, device, val_loader)
        # Collect some data for logging purposes.
        train_losses.append(float(train_loss))
        train_accs.append(train_acc)
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)

        print('\n\ttrain Loss: {:.6f}\ttrain acc: {:.6f} '
              '\n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(train_loss,
                                                               train_acc,
                                                               val_loss,
                                                               val_acc))

        # if epochStateSave is not None and epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"{epochStateSave}_epoch_{epoch}.pt")

    return [model, val_accs, val_losses, train_losses, train_accs]

models_for_experiment = [
    "CNN_2D_256"
    # "CNN_1D_RawMelSpectogram",
    # "CNN_1D_RawAudio",
    # "CNN_2D",
    # "BASIC_LSTM_BestAugmentation",
    # "RCNN_1D_BestAugmentation",
    # "RCNN_2D_BestAugmentation",
]

def main():

    for model_experiment_params in models_for_experiment:
        # modelToInstantiate = getModelFromModelName[model_experiment_params]
        modelToInstantiate = MelSpectogram_2DCNN

        modelsParentDirectory = os.path.abspath(os.path.join(modelToInstantiate.__file__, os.pardir))

        relativeParamDirectory = HParams(f"{modelsParentDirectory}/{LOCATION_TO_MODEL_PARAMS}/Models.yaml", model_experiment_params).path

        # print(yaml_map["CNN_1D_DefaultSpecs_Augmented"]["path"])
        params = HParams(f"{modelsParentDirectory}/{LOCATION_TO_MODEL_PARAMS}/{relativeParamDirectory}", model_experiment_params)

        dataset_path = f"{root_directory}/{params.data_dir}"
        
        runModelWithParams(modelToInstantiate,params,model_experiment_params,modelsParentDirectory,dataset_path)


if __name__ == '__main__':
    main()
