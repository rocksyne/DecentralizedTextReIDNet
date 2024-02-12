"""
ⓒ February 2023
Doc: Codebase configuration file.
License: Please see license file for details.
"""

# System modules import
import os
import platform

# 3rd party library
# from prettytable import PrettyTable

# Some global configurations
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
# table = PrettyTable(['Configuration Parameter','Value'], hrules=1)
# table.align['Configuration Parameter'] = 'l' # Left-align column
# table.align['Value'] = 'l' # Left-align the column


class DotDict(dict):
    """
    Doc.: dot.notation access to dictionary attributes
    Ref.: https://stackoverflow.com/a/23689767/3901871
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def sys_configuration(platform_name:str=platform.node(), dataset_name:str="CUHK-PEDES", show_configs:bool=False)->DotDict():
    """
    Doc.:   Provides dot.notation access to all system parameters
    Args.:  platform_name:  The name of the platform on which application is executing or training
                            Acceptable values are 'PC-SIM','deeplearning','ultron','nano'
            dataset_name:   The name of the dataset for training or evaluation
            show_configs:   Print out all the configurations or not
    Return: DotDict()
    """
    
    allowed_platforms_names:list[str] = ['PC-SIM','deeplearning','ultron','nano'] # add your platform name to this list
    allowed_dataset_names:list[str] = ["CUHK-PEDES","RSTPReid",'custom']

    if platform_name not in allowed_platforms_names:
        raise ValueError('`platform_name` be must string and of value `PC-SIM`,`deeplearning`,`ultron`,`nano` \
                         Provided name is `{}`. Add this name to the `allowed_platforms_names` variable.'.format(platform_name))
    
    if dataset_name not in allowed_dataset_names:
        raise ValueError('`dataset_name` be must string and of value `CUHK-PEDES`, `RSTPReid`\
                         Provided name is `{}`.'.format(dataset_name))
    
    # Experimented with different tokenizers to get the maximum index of the tokens
    # Bert tokenizer: 29609
    # Simple tokenizer: 49407
    # Tiktoken (cl100k_base): 100155
    # Tiktoken (p50k_base): 50276
    # Tiktoken (r50k_base): 50240
    max_tokenizer_size:dict[str,int] = dict(bert = 29609,
                                            simple_tokenizer = 49407,
                                            tiktoken_cl100k = 100155,
                                            tiktoken_p50k = 50276,
                                            tiktoken_r50k = 50240)
    
    configs = dict()

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++[Langauge model configurations]+++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['dropout_rate']:float = 0.30 # Probability one GRU neurns that should be dropped
    configs['num_layers']:int = 1 # Number of GRU layers
    configs["tokens_length_max"]:int = 100 # Maximum number of tokens
    configs['feature_length']:int = 1024 # Maximum number of features
    configs['tokenizer_type']:str = 'bert' # Tokenizer types: bert, simple_tokenizer, tiktoken_cl100k, tiktoken_p50k, tiktoken_r50k
    # get the appropriate vocabulary size relative the tokenizer type
    # Add +1 to account for array indexing related operations
    configs['vocab_size'] = max_tokenizer_size[configs['tokenizer_type']]+1 
    configs['embedding_dim']:int = 512 # number of embedding dimensions


    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++[visual model configuration]++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    configs["CUHKPEDES_image_size"]:tuple[int,int] = (384, 128) # (H,W)
    configs["MHPV2_image_size"]:tuple[int,int] = (473, 473) # (H,W) - https://github.com/liutinglt/CE2P/blob/master/dataset/datasets.py
                                                            # I think I will revert to (512,512) https://github.com/rocksyne/MHParsNet


    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++[Training configurations]+++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    configs['epoch']:int = 60# The umber of epochs for which model should be trained
    configs['adam_alpha']:float = 0.90 # Momentum term of AdamW
    configs['adam_beta']:float = 0.999 # Momentum term of AdamW
    configs["epoch_decay"]:list[int] = [20,40] # Params for torch.optim.lr_scheduler.StepLR: when decays should hapen
    configs['lr']:float = 0.001 # The initial learning rate
    configs['patience']:int = 3 # How many more epochs to train for after a metric fails to improve
    configs["val_dataset"]:str = "test" # Choose which dataset split to use for vaildation / evaluation. Values: `test` or `val`
    configs['device']:str  = "cuda" # If `cuda`, just use GPU ID 0 on all systems. Model is small anywa
    configs['seed']:int = 3407 # The seed for results reproduction
    configs['model_testing_data_split']:str = 'test' # The split of the dataset to use for testing. Values are `test` and `val`
    configs['save_best_test_results_only']:bool = True # Save the best test resuls or all results
    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++[Loss Functions Configurations]+++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    configs['margin']:float = 0.5
    configs['ranking_loss_alpha']:float = 1.0
    configs['identity_loss_beta']:float = 1.0


    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++[Data logging Configurations & Misc.]++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    configs['train_log_path'] = os.path.join(PARENT_DIR,'data','logs','train.log')
    configs['test_log_path']  = os.path.join(PARENT_DIR,'data','logs','test.log')
    configs['write_mode']:str = 'append' # Writing mode for log files. Values are `overwrite` and `append`
    configs['model_save_path']:str = os.path.join(PARENT_DIR, 'data', 'checkpoints')
    configs['plot_save_path']:str = os.path.join(PARENT_DIR, 'data', 'plots')
    configs['log_config_paprameters']:bool = True # Log config parameters as well
    configs['project_parent_dir']:str = PARENT_DIR # Absolute project path


    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++[Platform Specific Configurations]++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    if platform_name == 'PC-SIM': # PC dedicated simulation server
        configs['CUHK_PEDES_dataset_parent_dir']:str = "/home/users/roagyeman/research/datasets/CUHK-PEDES"
        configs['MHPv2_dataset_parent_dir']:str = "/home/users/roagyeman/research/datasets/LV-MHP-v2"
        configs['num_workers']:int = 16 # Use x CPU cores max
        configs['batch_size']:int = 10 # Self explanatory, but use x batches
        
    elif platform_name == 'deeplearning': # Development server
        configs['CUHK_PEDES_dataset_parent_dir']:str = "/media/rockson/Data_drive/datasets/CUHK-PEDES"
        configs['MHPv2_dataset_parent_dir']:str = "/media/rockson/Data_drive/datasets/LV-MHP-v2-bkup" # parent dir of the dataset
        configs['num_workers']:int = 6 # Use x CPU cores max
        configs['batch_size']:int = 10 # Self explanatory, but use x batches

    elif platform_name == 'ultron': # Other dedicated simulation server
        # configs['CUHK_PEDES_dataset_parent_dir']:str = "/datasets/CUHK-PEDES"
        # configs['num_workers'] = 6 # Use 6 CPU cores max
        # configs['batch_size']:int = 16 # Self explanatory, but use x batches
        ... # TODO
    
    elif platform_name == 'nano':
        ... # pass TODO: add nano specif configurations if needed
        
    else:
        raise ValueError("`{}` is an unknown platform name!")

    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++[Dataset Specific Configurations]++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    configs['dataset_name'] = dataset_name
    if configs["dataset_name"] == "CUHK-PEDES": # https://arxiv.org/pdf/1702.05729.pdf
        configs["dataset_path"]:str = configs['CUHK_PEDES_dataset_parent_dir']
        configs["mean"] = [0.485, 0.456, 0.406] # Mean for RGB channels
        configs["std"]  = [0.229, 0.224, 0.225] # Standard deviation for RGB channels

    elif configs["dataset_name"] == "RSTPReid": # https://arxiv.org/pdf/2109.05534.pdf
        raise NotImplementedError("No implementation for `{}` dataset.".format(configs["dataset_name"]))
    
    elif configs["dataset_name"] == "custom":
        configs["mean"] = [0.485, 0.456, 0.406] # Mean for RGB channels
        configs["std"]  = [0.229, 0.224, 0.225] # Standard deviation for RGB channels
    
    else:
        raise ValueError("Invalid value for dataset name.")
    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++[Debug Info Matters]+++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # if show_configs:
    #     for key, value in configs.items():
    #         table.add_row([key, value])
    #     print(table)

    return DotDict(configs)
