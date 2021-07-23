from ruamel.yaml import YAML

"""
I TOOK THIS FROM THE SAMPLE PROJECT GIVEN TO THE CLASS
"""

LOCATION_TO_MODEL_PARAMS="model_params"

class HParams(object):
    """
    .yaml file -> class representation
    """
    def __init__(self, yaml_file, model_name):
        with open(yaml_file) as f:
            yaml = YAML(typ='safe')
            # Load and find
            yaml_map = yaml.load(f)
            params_dict = yaml_map[model_name]
            for param in params_dict:
                setattr(self, param, params_dict[param])


def loadModelParameters(directory,modelName):
    relativeParamDirectory = HParams(f"{directory}/{LOCATION_TO_MODEL_PARAMS}/Models.yaml",
                                     modelName).path


    return HParams(f"{directory}/{LOCATION_TO_MODEL_PARAMS}/{relativeParamDirectory}",
                     modelName)
