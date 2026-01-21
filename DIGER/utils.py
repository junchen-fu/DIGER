import json
import datetime
import os
import pickle
import torch
import random
import faiss
import numpy as np
from copy import deepcopy
from collections import defaultdict
from torch import Tensor
import logging
import hashlib
from accelerate.utils import set_seed


def load_json(file: str):
    with open(file, 'r') as f:
        return json.loads(f.read())

def load_jsonl(file: str):
    load_data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            load_line = json.loads(line)
            load_data.append(load_line)
        return load_data

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")

    return cur

def get_seqs_len(seqs):
    seq_len = torch.tensor([len(seq) for seq in seqs])
    return seq_len

def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )

def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def safe_load(model, file, verbose=True):
    checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    
    # Auto-detect nested structure: if checkpoint has 'state_dict' key, extract it
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if verbose:
            print(f'## Detected nested checkpoint structure, extracting state_dict')
    else:
        state_dict = checkpoint
    
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    
    # Filter out EMA statistics and decoder keys from reporting (they're expected to be missing/extra)
    new_keys_in_new_filtered = [k for k in new_keys_in_new if not (k.startswith('decoder.') or '_ema' in k)]
    no_match_keys_filtered = [k for k in no_match_keys_of_model if not ('_ema' in k or 'N_ema' in k)]
    
    if verbose:
        print('##', model._get_name(), '# new keys in file:', new_keys_in_new_filtered, '# no match keys:', no_match_keys_filtered)
    model.load_state_dict(state_dict, strict=False)


def safe_load_embedding(model, file, verbose=True):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    if verbose:
        print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)

    matched_state_dict = deepcopy(model.state_dict())
    for key in model_state_dict_keys:
        if key in state_dict:
            file_size = state_dict[key].size(0)
            model_embedding = matched_state_dict[key].clone()
            model_size = model_embedding.size(0)
            model_embedding[:file_size, :] = state_dict[key][:model_size, :]
            matched_state_dict[key] = model_embedding
            if verbose:
                print(f'Copy {key} {matched_state_dict[key].size()} from {state_dict[key].size()}')
    model.load_state_dict(matched_state_dict, strict=False)


def norm_by_prefix(collection, prefix):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_collection = deepcopy(collection)
    global_mean = collection.mean(axis=0)
    global_var = collection.var(axis=0)
    for p, p_code in prefix_code.items():
        p_collection = collection[p_code]
        mean_value = p_collection.mean(axis=0)
        var_value = p_collection.var(axis=0)
        var_value[var_value == 0] = 1
        scale = global_var / var_value
        scale[np.isnan(scale)] = 1
        scale = 1
        p_collection = (p_collection - mean_value + global_mean) * scale
        new_collection[p_code] = p_collection
    return new_collection


def balance(code, ncentroids=10):
    num = [code.count(i) for i in range(ncentroids)]
    base = len(code) // ncentroids
    move_score = sum([abs(j - base) for j in num])
    score = 1 - move_score / len(code) / 2
    score = round(score, 4)
    return score


def conflict(code):
    code = [str(c) for c in code]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    max_value = max(list(freq_count.values()))
    min_value = min(list(freq_count.values()))
    len_set = len(set(code))
    return {'Max': max_value, 'Min': min_value, 'Type': len_set, '%': round(len_set / len(code), 4)}


def kmeans(x, ncentroids=10, niter=100):
    verbose = True
    x = np.array(x, dtype=np.float32)
    d = x.shape[1]
    model = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    model.train(x)
    D, I = model.index.search(x, 1)
    code = [i[0] for i in I.tolist()]
    return model.centroids, code


def add_last(file_in, code_num, file_out, cur_level):
    corpus_ids = json.load(open(file_in))
    docid_to_doc = defaultdict(list)
    new_corpus_ids = []
    for i, item in enumerate(corpus_ids):
        docid_to_doc[str(item)].append(i)
        new_corpus_ids.append(item + [len(docid_to_doc[str(item)]) % code_num + 1 + (cur_level-1)*code_num])
    json.dump(new_corpus_ids, open(file_out, 'w'))
    return new_corpus_ids


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice


def get_indices_count(all_indices_str):
    indices_count = defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


@torch.no_grad()
def sinkhorn_raw(out: Tensor, epsilon: float,
                 sinkhorn_iterations: int):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper

    B = Q.shape[1]
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1

    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)

    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.clamp(torch.sum(Q, dim=1, keepdim=True), min=1e-5)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(torch.sum(Q, dim=0, keepdim=True), dim=1, keepdim=True), min=1e-5)
        Q /= B
    Q *= B
    return Q.t()


def center_distance_for_constraint(distances):
    # distances: B, K
    max_distance = distances.max()
    min_distance = distances.min()

    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    assert amplitude > 0
    centered_distances = (distances - middle) / amplitude
    return centered_distances


def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config


def init_logger(config):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log')
    logfilepath = os.path.join(LOGROOT, config["dataset"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])


def init_seed(seed, reproducibility):
    r"""Initialize random seed for random functions in numpy, torch, cuda and cudnn.

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_file_name(config: dict, suffix: str = ''):
    config_str = "".join([str(value) for key, value in config.items() if (key != 'accelerator' and key != 'device') ])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    logfilename = "{}-{}{}".format(config['run_local_time'], md5, suffix)
    return logfilename


def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config


def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False  # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def parse_command_line_args(unparsed: list[str]) -> dict:

    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args


def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')

