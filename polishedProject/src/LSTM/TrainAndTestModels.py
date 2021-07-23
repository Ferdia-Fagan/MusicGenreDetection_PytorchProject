import os

from src.LSTM import BasicLSTM_1DCNN, BasicLSTM_2DCNN, RCNN_1D, RCNN_1D_GAP, RCNN_1D_WithConvsInsteadOfPools_WithGAP, \
    RCNN_2D, RawAudio_RCNN, RawAudio_RCNN_WithGAP, RCNN_1D_WithConvsInsteadOfPools, \
    RawAudio_RCNN_WithGAP_WithConvsInsteadOfMaxPools,BasicLSTM

from src.utils import HParams, Main



Main.setUpReproducibility()

# root_directory = f"{os.getcwd()}"
root_directory = os.path.abspath(os.path.dirname(__file__))

def runModel(model_name_testing, modelToTest):
    model_parameters_testing = HParams.loadModelParameters(root_directory,model_name_testing)

    datasetPath = f"{root_directory}/../../{model_parameters_testing.data_dir}"

    Main.runModelWithParams(modelToTest, model_parameters_testing,
                            model_name_testing, root_directory, datasetPath)

def run():
    # model_name_testing = "BASIC_LSTM_BestAugmentation"
    # runModel(model_name_testing, BasicLSTM)
    #
    # model_name_testing = "BASIC_LSTM_1DCNN_BestAugmentation"
    # runModel(model_name_testing, BasicLSTM_1DCNN)
    #
    # model_name_testing = "BASIC_LSTM_2DCNN_BestAugmentation"
    # runModel(model_name_testing, BasicLSTM_2DCNN)
    #
    # model_name_testing = "RCNN_1D_BestAugmentation"
    # runModel(model_name_testing, RCNN_1D)
    #
    # model_name_testing = "RCNN_1D_Gap"
    # runModel(model_name_testing, RCNN_1D_GAP)
    #
    # model_name_testing = "RCNN_1D_WithConvsInsteadOfPools"
    # runModel(model_name_testing, RCNN_1D_WithConvsInsteadOfPools)
    #
    # model_name_testing = "RCNN_1D_WithConvsInsteadOfPools_WithGAP"          # todo:
    # runModel(model_name_testing, RCNN_1D_WithConvsInsteadOfPools_WithGAP)
    #
    # model_name_testing = "RCNN_2D_BestAugmentation"
    # runModel(model_name_testing, RCNN_2D)
    #
    # model_name_testing = "RCNN_1D_RawAudio"
    # runModel(model_name_testing, RawAudio_RCNN)
    #
    # model_name_testing = "RCNN_1D_RawAudio_WithGAP"
    # runModel(model_name_testing, RawAudio_RCNN_WithGAP)

    model_name_testing = "RCNN_1D_RawAudio_WithGAP_WithConvsInsteadOfMaxPools"                             # todo:
    runModel(model_name_testing, RawAudio_RCNN_WithGAP_WithConvsInsteadOfMaxPools)

run()