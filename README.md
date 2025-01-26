# TUTNet
torch >= 2.0.0
cuda >= 11.8

from MYNetpro import MYNet
from TF_configspro import get_model_config

if model_type == "MYNet":
        config_vit = get_model_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer head dim: {}'.format(config_vit.transformer.embedding_channels))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = MYNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels, img_size=config.img_size) 
