# Import libraries for process config YAML File
import yaml

# Import other libraries
import os


CONFIG_PATH = "conf\\"

def load_config(config_name):
    """"""
    try:
        with open(os.path.join(CONFIG_PATH, config_name), "r") as conf_file:
            config = yaml.safe_load(conf_file)
    except Exception as e:
        print("Error reading the config file")

    return config
