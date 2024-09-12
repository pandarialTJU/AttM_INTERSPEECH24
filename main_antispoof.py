"""
Code for INTERSPEECH 2024 paper: Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection

Pan, Z., Liu, T., Sailor, H.B., Wang, Q. (2024) Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection. Proc. Interspeech 2024, 2090-2094, doi: 10.21437/Interspeech.2024-1472

@inproceedings{pan24c_interspeech,
  title     = {Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection},
  author    = {Zihan Pan and Tianchi Liu and Hardik B. Sailor and Qiongqiong Wang},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2090--2094},
  doi       = {10.21437/Interspeech.2024-1472},
}

"""
___author__ = "Pan Zihan"
__email__ = "talpanzh@gmail.com"

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (CyberDataset,
                        CyberEvalDataset,
                        gen_cyber_list,
                        CyberPatialDataset,
                        RandomLengthDataLoader)
from evaluation import calculate_tDCF_EER, compute_nist_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# # Define the list of device IDs to be used for computations
num_GPU = 1

all_device = [0,1,2,3,4,5,6,7]
device_ids = all_device[0:num_GPU]

torch.autograd.set_detect_anomaly(True)

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    # model_config["model_name"] = "Model"

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    fold_id = args.fold
    meta_path = Path(args.meta_dir)
    feat_file = Path(args.feat_file)

    # no need to use it, if we have the tsv file path.
    trn_list_path = (meta_path / f"fold{fold_id}_train.tsv")
    dev_trial_path = (meta_path / f"fold{fold_id}_validation.tsv")
    eval_trial_path = (meta_path / f"fold{fold_id}_evaluation.tsv")


    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag    
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture

    model = get_Wavlm_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(feat_file, trn_list_path,
                                                     dev_trial_path, eval_trial_path,
                                                     args.seed, config)
    


    # evaluation on a pretrained checkpoint
    if args.eval: 
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        
        
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path)
        compute_nist_eer(sc_file=eval_score_path,
                         output_file=model_tag / "EER.txt")
        print("DONE.")
        sys.exit(0)

    # load model state for tuning
    if args.tune:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start tuning...")



    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)


    best_dev_eer = 100.
    best_eval_eer = 100.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    save_all = True

    # Training
    loss_history = []
    dev_eer_history = []

    params_backend = list(model.module.featfusion.parameters()) + list(model.module.decoder.parameters())
    optimizer1, scheduler1= create_optimizer(params_backend, optim_config)
    optimizer2, scheduler2= create_optimizer(model.module.parameters(), optim_config)

    best_loss = float('inf')
    patience = 5
    minimum_epochs = 20


    for epoch in range(config["num_epochs"]):


        if epoch < config["tune_start_epoch"]:
            optimizer = optimizer1
            scheduler = scheduler1
        else:
            optimizer = optimizer2
            scheduler = scheduler2


        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                scheduler, config)

        valid_loss = produce_evaluation_file(dev_loader, model, device,
                                metric_path / "dev_score.txt")
        dev_eer, dev_th = compute_nist_eer(sc_file=metric_path / "dev_score.txt",
                                        output_file=metric_path / "dev_EER_{}epo.txt".format(epoch))
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_th:{:.5f}".format(
            running_loss, dev_eer, dev_th))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_th", dev_th, epoch)


        scheduler1.step()
        scheduler2.step()


        # Save the loss for this epoch to a text file
        loss_history.append(running_loss)
        dev_eer_history.append(dev_eer)

        
        if best_dev_eer >= dev_eer or save_all:
            torch.save(model.state_dict(),
                    model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))


        writer.add_scalar('Loss/train', running_loss, epoch)
        current_lr = scheduler.get_lr()
        writer.add_scalar('Learning Rate', current_lr[0], epoch)
        writer.add_scalar('Loss/validation', valid_loss, epoch)


        with open(model_tag/'loss_history.txt', 'a') as file:
            file.write(f'Epoch {epoch + 1}: {running_loss}\n')
        with open(model_tag/'validation_eer_history.txt', 'a') as file:
            file.write(f'Epoch {epoch + 1}: {dev_eer}\n')

        with open(model_tag/'learning_rate.txt', 'a') as file:
            file.write(f'Epoch {epoch + 1}: {current_lr[0]}\n')
        with open(model_tag/'valid_loss.txt', 'a') as file:
            file.write(f'Epoch {epoch + 1}: {valid_loss}\n')

        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_no_improve = 0
            print('best valid loss:', best_loss)
        else:
            epochs_no_improve += 1
            print('best valid loss:', best_loss)

        # Apply early stopping after minimum_epochs
        if epoch >= minimum_epochs and epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

####################### average the 5 checkpoints ##############################
            
    # Path to your validation loss text file
    txt_path = model_tag/'valid_loss.txt'
    losses = read_losses(txt_path)

    # Sort losses and get the epochs with the 5 minimum validation losses
    min_epochs = sorted(losses, key=losses.get)[:5]

    # Path to the directory containing checkpoints
    checkpoint_dir = model_save_path

    # Generate checkpoint file names
    # Assuming the metric value in the filename isn't needed to identify the checkpoint
    checkpoint_paths = [os.path.join(checkpoint_dir, f'epoch_{epoch}_*.pth') for epoch in min_epochs]

    # Find the actual file names (since the metric value is unknown)
    import glob
    actual_checkpoint_paths = []
    for path_pattern in checkpoint_paths:
        # This will get the first file matching the pattern
        actual_checkpoint_paths.extend(glob.glob(path_pattern))

    # Average the checkpoints
    avg_checkpoint = average_checkpoints_debug(actual_checkpoint_paths)

    best_checkpoint_path = checkpoint_dir/'averaged_checkpoint.pth'

    # Saving the averaged checkpoint
    torch.save(avg_checkpoint, best_checkpoint_path)
            


####################################### valid and eval 2019 LA ############################
    
    model.load_state_dict(
            torch.load(best_checkpoint_path, map_location=device))

    
    valid_loss = produce_evaluation_file(dev_loader, model, device,
                            metric_path / "RE_dev_score.txt")
    valid_eer, _ = compute_nist_eer(sc_file=metric_path / "RE_dev_score.txt",
                                    output_file=metric_path / "valid2019_best.txt")  

            

    eval_score_path = model_tag / 'eval_2019_output.txt'
    eval_loss = produce_evaluation_file(eval_loader, model, device,
                            eval_score_path)
    eval_eer, _ = compute_nist_eer(sc_file=eval_score_path,
                                output_file=metric_path / "eval2019_best.txt")




def get_Wavlm_model(model_config: Dict, device: torch.device):
        
        module = import_module("models.{}".format(model_config["architecture"]))
        _model = getattr(module, model_config["model_name"])
       
        # for evalution experiment 15, 16, 17 for wav2vec2_finetune_all checkpoints, need to 

        from WavLM import WavLM, WavLMConfig


        # put the pre-trained Wavlm_Large model checkpoint here
        checkpoint = torch.load('your/path/to/WavLM-Large.pt')

        cfg = WavLMConfig(checkpoint['cfg'])
        speech_model = WavLM(cfg)
        speech_model.load_state_dict(checkpoint['model'])

        speech_model.encoder.layers = torch.nn.Sequential(*[speech_model.encoder.layers[i] for i in range(model_config["SSL_layer_num"])])

        # model_config["SSL_layer_num"]

        model = _model(speech_model, model_config)
        model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. model params:{}".format(nb_params))

        return model




def get_loader(
        feat_file: str,
        trn_list_path: str,
        dev_trial_path: str,
        eval_trial_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    if os.path.exists(trn_list_path):
        trn_keys, trn_labs, trn_paths = gen_cyber_list(meta_file=trn_list_path,
                                                       feat_file=feat_file)
        print("no. training files:", len(trn_keys))

        train_set = CyberDataset(list_ids=trn_keys,
                                 labels=trn_labs,
                                 file_paths=trn_paths)
        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen,num_workers=16)
    else:
        print('[WARNING] no training file list, it is possible only for evaluation case.')
        trn_loader = None

    if os.path.exists(dev_trial_path):
        dev_keys, dev_labs, dev_paths = gen_cyber_list(meta_file=dev_trial_path,
                                                       feat_file=feat_file)
        print("no. validation files:", len(dev_keys))

        dev_set = CyberEvalDataset(list_ids=dev_keys,
                                   labels=dev_labs,
                                   file_paths=dev_paths)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,num_workers=16)
    else:
        print('[WARNING] no dev file list, it is possible only for evaluation case.')
        dev_loader = None

    eval_keys, eval_labs, eval_paths = gen_cyber_list(meta_file=eval_trial_path,
                                                      feat_file=feat_file)
    eval_set = CyberEvalDataset(list_ids=eval_keys,
                                labels=eval_labs,
                                file_paths=eval_paths)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,num_workers=16)

    return trn_loader, dev_loader, eval_loader




def produce_evaluation_file(
        data_loader: DataLoader,
        model,
        device: torch.device,
        save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    lab_list = []

        # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    valid_loss = 0.0
    num_total = 0.0

    for i, (batch_x, batch_y, utt_id) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        with torch.no_grad():
            # _, batch_out = model(batch_x) # for AASIST
            batch_out = model(batch_x) # for wav2vec_AASIST
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() # 1 - detect bona, 0 - detect spoof

            batch_loss = criterion(batch_out, batch_y)
            valid_loss = valid_loss + batch_loss.item()*batch_size
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lab_list.extend(batch_y)
        #print(i, utt_id)

    with open(save_path, "w") as fh:
        for fn, lab, sco in zip(fname_list, lab_list, score_list):
            lab = "bonafide" if lab == 1 else "spoof"
            fh.write(f"{fn}\t{lab}\t{sco}\n")
    print("Scores saved to {}".format(save_path))

    valid_loss /= num_total

    return valid_loss


# Function to read validation losses from the text file
def read_losses(file_path):
    losses = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch'):
                parts = line.split(':')
                epoch = int(parts[0].split()[1])  # Adjusting for zero-indexing
                loss = float(parts[1].strip())
                losses[epoch] = loss
    return losses


def train_epoch(
        trn_loader: DataLoader,
        model,
        optim: Union[torch.optim.SGD, torch.optim.Adam],
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, utt_id in trn_loader:

        optim.zero_grad()

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"])) # for aasist
        batch_out = model(batch_x) #for wav2vec aasist
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        
        batch_loss.backward()
        optim.step()


    running_loss /= num_total
    return running_loss


def average_checkpoints_debug(checkpoint_paths):
    # Load all checkpoints and store their state_dicts
    state_dicts = [torch.load(path) for path in checkpoint_paths]

    # Initialize a dictionary to store the averaged parameters
    avg_state_dict = {key: torch.zeros_like(state_dicts[0][key]) for key in state_dicts[0]}

    # Sum and average the parameters
    for key in state_dicts[0]:
        # Convert to float, sum and average the parameters
        avg_state_dict[key] = sum([state_dict[key].float() for state_dict in state_dicts]) / len(state_dicts)

        # Convert back to original data type if necessary
        if state_dicts[0][key].dtype != torch.float32:
            avg_state_dict[key] = avg_state_dict[key].type(state_dicts[0][key].dtype)

    return avg_state_dict


if __name__ == "__main__":

    # torch.cuda.empty_cache()
    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )  
    parser.add_argument(
        "--meta_dir",
        dest="meta_dir",
        type=str,
        help="processed meta files following cyber_cookies format",
        default="./data/meta/",
    )
    parser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        help="fold number",
        default=1,
    )
    parser.add_argument(
        "--feat_file",
        dest="feat_file",
        type=str,
        help="file with all features, follows cyber_cookies format (wav.scp)",
        default="./data/meta/wav.scp",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--SSL_num",
                        type=int,
                        default=12,
                        help="number of the layers in SSL model")    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--pretrain_checkpoint",
                        type=str,
                        default=None,
                        help="the checkpoint path")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_path",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument(
        "--num_gpu",
        action="store_true",
        help="when this flag is given, continue train the model from pre-trained checkpoint")
    main(parser.parse_args())
