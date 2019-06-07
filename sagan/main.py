from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder

def main(config):
    # For fast training
    cudnn.benchmark = False


    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
       pass
if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)