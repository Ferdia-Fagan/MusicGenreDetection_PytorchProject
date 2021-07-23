import os

from src.CNN_2D import MelSpectogram_CNN_WithGAP, MelSpectogram_CNN, MelSpectogram_CNN_CurringOverfitting
from src.utils import HParams, Main

# root_directory = f"{os.getcwd()}"
root_directory = os.path.abspath(os.path.dirname(__file__))

def runModel(model_name_testing, modelToTest):
    model_parameters_testing = HParams.loadModelParameters(root_directory,model_name_testing)

    datasetPath = f"{root_directory}/../../{model_parameters_testing.data_dir}"

    Main.runModelWithParams(modelToTest, model_parameters_testing,
                            model_name_testing, root_directory, datasetPath)

# -----------------------------------------------------------
def run():
    model_name_testing = "CNN_2D_512"
    runModel(model_name_testing,MelSpectogram_CNN)

    model_name_testing = "CNN_2D_512_CurringOverfitting"        # todo:
    runModel(model_name_testing,MelSpectogram_CNN_CurringOverfitting)

    model_name_testing = "CNN_2D_512_SmallerConvolutions" # far worse
    runModel(model_name_testing,MelSpectogram_CNN)

    model_name_testing = "CNN_2D_512_WithGAP"
    runModel(model_name_testing,MelSpectogram_CNN_WithGAP)

# run()



