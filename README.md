# TUTNet
torch >= 2.0.0
cuda >= 11.8

from MYNetpro import MYNet
from TF_configspro import get_model_config

config_vit = get_model_config()
model = MYNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels, img_size=config.img_size) 
