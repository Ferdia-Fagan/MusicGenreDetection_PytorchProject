
def getParamsForModel(root_directory, modelName):
    return f"{root_directory}/ModelStates/{modelName}/{modelName}.pt"

def getLogsForModel(root_directory, modelName):
    return f"{root_directory}/logs/{modelName}.json"
