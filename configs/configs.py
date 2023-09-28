import yaml

def load_configs():
    with open("./configs/configs.yaml") as stream:
        configs: dict = yaml.safe_load(stream)
    return configs