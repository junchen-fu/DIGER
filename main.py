import yaml
import argparse
import warnings
import torch
from data import load_split_data
from data import SequentialSplitDataset, Collator
from torch.utils.data import DataLoader
from trainer import Trainer
from transformers import T5Config, T5ForConditionalGeneration
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list
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
    batch_info = get_training_batch_info(
        config['batch_size'], accelerator.num_processes,
        config['gradient_accumulation_steps']
    )
    log(
        '[Batch] per_device={per_device_batch_size}, processes={num_processes}, '
        'accumulation={gradient_accumulation_steps}, effective={effective_batch_size}'.format(**batch_info),
        accelerator, logger
    )
    reference_batch_size = config.get('reference_effective_batch_size')
    if reference_batch_size and batch_info['effective_batch_size'] != reference_batch_size:
        log(
            f'[Batch] Effective batch size {batch_info["effective_batch_size"]} differs from '
            f'the paper setting {reference_batch_size}.',
            accelerator, logger, level='warning'
        )
    log(f'Config: {str(config)}', accelerator, logger)

    item2id, num_items, train, valid, test = load_split_data(config)
    code_num = config['code_num']
    code_length = config['code_length']                             
    eos_token_id = -1
    batch_size=config['batch_size']
    eval_batch_size=config['eval_batch_size']
    
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)
    semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
    
    
    accelerator.wait_for_everyone()
                                                        
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
    if config.get('gradient_checkpointing', False):
        t5.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        t5.config.use_cache = False
        log('[Memory] T5 gradient checkpointing enabled', accelerator, logger)
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
        
                                                                            
                                                                           
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            use_simple_uncertainty = config.get('use_simple_uncertainty_loss', False)
            
            if use_simple_uncertainty:
                                                                            
                target_sigma = initial_std
                log(f"[Config] Resetting sigma to {target_sigma} (direct std, from initial_std={initial_std})", accelerator, logger)
            else:
                                                    
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
                if 'sigma' in name:                                                                
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

    process_seeds = accelerator.gather(
        torch.tensor([trainer.process_seed], device=accelerator.device, dtype=torch.long)
    )
    log(
        f'[Seed] per_process_cuda={process_seeds.cpu().tolist()}',
        accelerator,
        logger,
    )

    best_score = trainer.train(verbose=verbose)
    test_results = trainer.test()

    if accelerator.is_main_process:
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)
    accelerator.end_training()


if __name__=="__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)

            
    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)

    config = convert_config_dict(config)
    gradient_accumulation_steps = int(config.get('gradient_accumulation_steps', 1))
    if gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be at least 1')

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    dataset = config['dataset']
    config['device'] = accelerator.device
    config['use_ddp'] = accelerator.num_processes > 1
    run_local_time = [get_local_time() if accelerator.is_main_process else None]
    broadcast_object_list(run_local_time)
    config['run_local_time'] = run_local_time[0]

    ckpt_name = get_file_name(config)

    config['save_path'] =f'./myckpt/{dataset}/{ckpt_name}'
    
    config['accelerator'] = accelerator
    
        
    train(config, verbose=accelerator.is_main_process, rank=accelerator.process_index)

    
    
