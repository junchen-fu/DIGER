import yaml
import argparse
import warnings
import torch
from data import load_split_data
from data import SequentialSplitDataset, Collator
from torch.utils.data import DataLoader
from trainer import Trainer
from transformers import T5Config, T5ForConditionalGeneration
from accelerate import Accelerator
from model import Model
from utils import *
from vq import RQVAE
from logging import getLogger
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="./config/scientific.yaml")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def train(config, verbose=True, rank=0):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    accelerator = config['accelerator']
    
    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)

    item2id, num_items, train, valid, test = load_split_data(config)
    code_num = config['code_num']
    code_length = config['code_length'] # current length of the code
    eos_token_id = -1
    batch_size=config['batch_size']
    eval_batch_size=config['eval_batch_size']
    
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)
    semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
    
    
    accelerator.wait_for_everyone()
    # Initialize the model with the custom configuration
    model_config = T5Config(
            num_layers=config['encoder_layers'], 
            num_decoder_layers=config['decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=1,
            pad_token_id=0,
            eos_token_id=300,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=config['max_length'],
        )
    
    t5 = T5ForConditionalGeneration(config=model_config)
    model_rec = Model(config=config, model=t5, n_items=num_items,
                  code_length=code_length, code_number=code_num)
    

    semantic_emb = np.load(semantic_emb_path)
        
    model_rec.semantic_embedding.weight.data[1:] = torch.tensor(semantic_emb).to(config['device'])
    model_id = RQVAE(config=config, in_dim=model_rec.semantic_hidden_size)
    
    log(model_rec, accelerator, logger)
    log(model_id, accelerator, logger)

    rqvae_path = config.get('rqvae_path', None)
    if rqvae_path is not None:
        safe_load(model_id, rqvae_path, verbose)
        
        # NEW: Force reset sigma if initial_sigma OR initial_std is provided
        # This prevents checkpoint from overwriting the desired noise level
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            use_simple_uncertainty = config.get('use_simple_uncertainty_loss', False)
            
            if use_simple_uncertainty:
                # AutoSigmaSimple: sigma is directly the std (no conversion)
                target_sigma = initial_std
                log(f"[Config] Resetting sigma to {target_sigma} (direct std, from initial_std={initial_std})", accelerator, logger)
            else:
                # AutoSigmaGumbel: sigma = log2(std)
                if initial_std <= 1e-5:
                    target_sigma = -20.0
                else:
                    import math
                    target_sigma = math.log2(initial_std)
                log(f"[Config] Resetting sigma to {target_sigma} (log2 space, from initial_std={initial_std})", accelerator, logger)
            
            count = 0
            for name, param in model_id.named_parameters():
                if 'sigma' in name:
                    param.data.fill_(target_sigma)
                    count += 1
            log(f"  -> Reset {count} sigma parameters.", accelerator, logger)
            
        elif 'initial_sigma' in config:
            initial_sigma = float(config['initial_sigma'])
            log(f"[Config] Resetting sigma to {initial_sigma} (overriding checkpoint)", accelerator, logger)
            count = 0
            for name, param in model_id.named_parameters():
                if 'sigma' in name: # and param.requires_grad: # Reset even if frozen, just in case
                    param.data.fill_(initial_sigma)
                    count += 1
            log(f"  -> Reset {count} sigma parameters.", accelerator, logger)

    train_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=train)
    valid_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=valid)
    test_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=test)

    collator = Collator(eos_token_id=eos_token_id, pad_token_id=0, max_length=config['max_length'])

    train_data_loader = DataLoader(train_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    
    
    trainer = Trainer(config=config, model_rec=model_rec, model_id=model_id, accelerator=accelerator, train_data=train_data_loader,
                      valid_data=valid_data_loader, test_data=test_data_loader, eos_token_id=eos_token_id)

    # Train with frozen RQ-VAE
    best_score = trainer.train(verbose=verbose)
    test_results = trainer.test()

    if accelerator.is_main_process:
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)


if __name__=="__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Config
    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset = config['dataset']
    
    local_time = get_local_time()
    config['device'], config['use_ddp'] = init_device()
    accelerator = Accelerator()

    # gather all the config and set the checkpoint name
    all_run_local_time = accelerator.gather_for_metrics([local_time])
    config['run_local_time'] = all_run_local_time[0]

    ckpt_name = get_file_name(config)

    config['save_path'] =f'./myckpt/{dataset}/{ckpt_name}'
    
    config = convert_config_dict(config)
    config['accelerator'] = Accelerator()
    
        
    train(config, verbose=local_rank==0, rank=local_rank)

    

    
    