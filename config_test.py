import importlib.util as imp
import sys

config_module = "configs.default_config"
config_filepath = r".\laparoscopic_segmentation\configs\default_config.py"

# import config
spec = imp.spec_from_file_location(config_module, config_filepath)
config = imp.module_from_spec(spec)
sys.modules[config_module] = config
spec.loader.exec_module(config)
conf = config.Config()

print(conf)

pass
