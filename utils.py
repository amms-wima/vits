import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import datetime
import shutil

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def query_checkpoint(checkpoint_path):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  try:
    checkpoint_dict['optimizer']
    has_opt = True
  except:
    has_opt = False
  try:
    best_losses = checkpoint_dict['best_losses']
    bl_loaded = True
  except:
    best_losses = None
  try:
    gbl_step = checkpoint_dict['gbl_step']
  except:
    gbl_step = 0
  logger.info(f"Loaded '{checkpoint_path}' ({gbl_step}/{iteration}) [opt:{has_opt}, bl:{bl_loaded}]")
  if (bl_loaded):
    logger.info(f"\tlr: {best_losses}")
  logger.info(f"\tlr: {learning_rate}")
  saved_state_dict = checkpoint_dict['model']
  if "enc_p.emb.weight" in saved_state_dict:
    num_rows_enc_checkpoint = saved_state_dict["enc_p.emb.weight"].shape[0]
    logger.info(f"\tenc_p.emb.weight rows: {num_rows_enc_checkpoint}")
  if "emb_g.weight" in saved_state_dict:
    num_rows_emb_checkpoint = saved_state_dict["emb_g.weight"].shape[0]
    logger.info(f"\temb_g.weight rows: {num_rows_emb_checkpoint}")
    

def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration, _ = _load_checkpoint_dict_entry(checkpoint_dict, 'iteration', raise_if_not_found=True)
  learning_rate, _ = _load_checkpoint_dict_entry(checkpoint_dict, 'learning_rate', raise_if_not_found=True)
  best_losses, bl_loaded = _load_checkpoint_dict_entry(checkpoint_dict, 'best_losses', None)
  gbl_step, _ = _load_checkpoint_dict_entry(checkpoint_dict, 'gbl_step', 0)

  saved_state_dict = checkpoint_dict['model']
  is_model_module = hasattr(model, 'module')
  state_dict = model.module.state_dict() if is_model_module else model.state_dict()
  mdl = model.module if is_model_module else model
  idx_diff_dict = _realign_weight_tensor_size_where_reqd(saved_state_dict, state_dict)
  opt_loaded = _load_optimizer(optimizer, checkpoint_dict, idx_diff_dict)

  _load_dict_key_values(mdl, saved_state_dict, state_dict)
  logger.info(f"Loaded '{checkpoint_path}' ({gbl_step}/{iteration}) [opt:{opt_loaded}, bl:{bl_loaded}]")
  return model, optimizer, learning_rate, iteration, gbl_step, best_losses


def _load_checkpoint_dict_entry(checkpoint_dict, key, default_val=None, raise_if_not_found=False):
  try:
    val = checkpoint_dict[key]
    loaded = True
  except KeyError as e:
    if raise_if_not_found:
      raise e
    val = default_val
    loaded = False
  return val, loaded


def _load_dict_key_values(model, saved_state_dict, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
      try:
        new_state_dict[k] = saved_state_dict[k]
      except:
        logger.info(f"{k} is not in the checkpoint")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def _load_optimizer(optimizer, checkpoint_dict, idx_diff_dict):
  if optimizer is not None:
    try:
      saved_state_dict = checkpoint_dict['optimizer']
      _realign_optimizer_tensor_size_where_reqd(saved_state_dict, idx_diff_dict)
      optimizer.load_state_dict(saved_state_dict)
      opt_loaded = True
    except:
      logger.warning("optimizer was not loaded from checkpoint dictionary", sys.exc_info())
  return opt_loaded


def _realign_weight_tensor_size_where_reqd(saved_state_dict, state_dict):
  idx = 0
  idx_diff_dict = {}
  for key in saved_state_dict:
    if key in state_dict and saved_state_dict[key].shape != state_dict[key].shape:
      num_rows_current_model = state_dict[key].shape[0]
      diff_rows = state_dict[key].shape[0] - saved_state_dict[key].shape[0]
      if (diff_rows > 0):
        idx_diff_dict[idx] = diff_rows
        saved_state_dict[key] = torch.cat((saved_state_dict[key], torch.zeros((diff_rows, saved_state_dict[key].shape[1]))), dim=0)[:num_rows_current_model, :]
    idx += 1
  return idx_diff_dict


def _realign_optimizer_tensor_size_where_reqd(saved_state_dict, idx_diff_dict):
  for key, diff_rows in idx_diff_dict.items():
    if 'exp_avg' in saved_state_dict['state'][key]:
      saved_exp_avg = saved_state_dict['state'][key]['exp_avg']
      padding = torch.zeros((diff_rows, saved_exp_avg.shape[1]), dtype=saved_exp_avg.dtype)
      saved_state_dict['state'][key]['exp_avg'] = torch.cat((saved_exp_avg, padding), dim=0)

      if 'exp_avg_sq' in saved_state_dict['state'][key]:
        saved_exp_avg_sq = saved_state_dict['state'][key]['exp_avg_sq']
        padding = torch.zeros((diff_rows, saved_exp_avg_sq.shape[1]), dtype=saved_exp_avg_sq.dtype)
        saved_state_dict['state'][key]['exp_avg_sq'] = torch.cat((saved_exp_avg_sq, padding), dim=0)


def sync_checkpoint(checkpoint, prev_cp, hps):
  sync_latest_cp_path = os.path.join(hps.model_dir, checkpoint)
  sync_checkpoint_path = os.path.join(hps.model_sync_folder, checkpoint)
  sync_prev_cp_path = os.path.join(hps.model_sync_folder, prev_cp)
  if os.path.exists(sync_checkpoint_path):
      shutil.move(sync_checkpoint_path, sync_prev_cp_path)
  shutil.copy2(sync_latest_cp_path, sync_checkpoint_path)


def save_if_best_model(curr_losses, best_losses, epoch, hps, net_g, net_d, optim_g, optim_d, gbl_step):
  if best_losses is None or sum(curr_losses) < sum(best_losses):
    best_losses = curr_losses
    hps.in_train_manifest["best_model"]["iteration"] = epoch
    hps.in_train_manifest["best_model"]["step"] = gbl_step
    hps.in_train_manifest["best_model"]["losses"] = best_losses
    save_checkpoints(epoch, hps, net_g, net_d, optim_g, optim_d, gbl_step, True)
  return best_losses


def save_checkpoints(epoch, hps, net_g, net_d, optim_g, optim_d, gbl_step, is_best=False):
    prefix = "^" if is_best else "_"
    save_in_train_manifest(hps, epoch, gbl_step, False)
    save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, f"G{prefix}latest.pth", gbl_step, hps)
    save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, f"D{prefix}latest.pth", gbl_step, hps)
    save_in_train_manifest(hps, epoch, gbl_step, True)


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint, gbl_step=0, hps=None):
  prev_cp = checkpoint.replace("latest", "previous")
  checkpoint_path = os.path.join(hps.model_dir, checkpoint)
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  prev_cp_path = os.path.join(hps.model_dir, prev_cp)
  if (hps.save_prev_backup and os.path.exists(checkpoint_path)):
      shutil.move(checkpoint_path, prev_cp_path)  
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate,
              'best_losses': hps.in_train_manifest["best_model"]["losses"],
              'gbl_step': gbl_step,
              }, checkpoint_path)
  if (hps.model_sync_folder is not None):
    sync_checkpoint(checkpoint, prev_cp, hps)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-o', '--output_path', type=str, required=True,
                      help='training output directory')
  parser.add_argument('-rg', '--restore_gen_file', type=str, default=None,
                      help='restore generator model file to continue training')
  parser.add_argument('-rd', '--restore_dis_file', type=str, default=None,
                      help='restore discriminator model file to continue training')
  parser.add_argument('-fl', '--freeze_layers', type=int, default=0, 
                      help='set layers to be frozen when fine-tuning')
  parser.add_argument('-lo', '--load_optimisation', type=int, default=1, 
                      help='loads the optimisation in utils.load_checkpoint()')
  parser.add_argument('-rlroe', '--reset_learning_rate_optimiser_epoch', type=int, default=0, 
                      help='uses -1 for torch.optim.lr_scheduler.ExponentialLR if set')
  parser.add_argument('-spb', '--save_prev_backup', type=int, default=1, 
                      help='set to 0 only when using an external backup utility otherwise ?_latest.pth will not be saved to ?_previous.pth')
  parser.add_argument('-s', '--start_global_step', type=int, default=-1, help='start global steps count [-1=system determined]')
  parser.add_argument('-msf', '--model_sync_folder', type=str, default=None, help='sync the model files to a sync folder (eg. /content/drive/MyDrive/vits/build)')
  
  args = parser.parse_args()
  output_path, config = _read_config_and_save_output_path(init, args)
  
  hparams = HParams(**config)
  hparams.model_dir = output_path
  if (args.restore_gen_file is None): 
    hparams.restore_gen_file = os.path.join(hparams.model_dir, "G^latest.pth")
  if (args.restore_dis_file is None): 
    hparams.restore_dis_file = os.path.join(hparams.model_dir, "D^latest.pth")
  hparams.freeze_layers = args.freeze_layers == 1
  hparams.load_optimisation = args.load_optimisation == 1
  hparams.reset_learning_rate_optimiser_epoch = args.reset_learning_rate_optimiser_epoch == 1
  hparams.start_global_step = args.start_global_step
  hparams.model_sync_folder= args.model_sync_folder
  hparams.save_prev_backup= args.save_prev_backup == 1

  _initialise_in_train_manifest_with_default_values(output_path, hparams)
  return hparams


def _read_config_and_save_output_path(init, args):
    output_path = os.path.join("./", args.output_path)

    if not os.path.exists(output_path):
      os.makedirs(output_path)

    config_path = args.config
    config_save_path = os.path.join(output_path, "config.json")
    if init:
      with open(config_path, "r") as f:
        data = f.read()
      with open(config_save_path, "w") as f:
        f.write(data)
    else:
      with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    return output_path, config


def _initialise_in_train_manifest_with_default_values(output_path, hparams):
    hparams.in_train_manifest_path = os.path.join(output_path, "in_train_manifest.json")
    hparams.in_train_manifest = {}
    hparams.in_train_manifest["last_updated"] = "<pending>"
    hparams.in_train_manifest["abort_on_next_iteration"] = False
    hparams.in_train_manifest["global_step_start"] = -1
    hparams.in_train_manifest["iteration"] = -1
    hparams.in_train_manifest["previous_step"] = -1
    hparams.in_train_manifest["latest_step"] = -1  
    hparams.in_train_manifest["after_save"] = False  
    hparams.in_train_manifest["chg_eval_interval"] = -1
    hparams.in_train_manifest["best_model"] = {}
    hparams.in_train_manifest["best_model"]["iteration"] = -1
    hparams.in_train_manifest["best_model"]["step"] = -1
    hparams.in_train_manifest["best_model"]["losses"] = None


def _serialize_tensor(tensor):
  return tensor.item() if tensor.numel() == 1 else tensor.tolist()


def _deserialize_tensor(serialized_tensor):
  if isinstance(serialized_tensor, (int, float)):
      return torch.tensor(serialized_tensor)
  elif isinstance(serialized_tensor, list):
      return torch.tensor(serialized_tensor)
  else:
      raise ValueError("Invalid serialized tensor format")


def reload_in_train_manifest(hps):
    if os.path.exists(hps.in_train_manifest_path):
        with open(hps.in_train_manifest_path, "r") as f:
            data = f.read()
            config = json.loads(data)
            if ("abort_on_next_iteration" in config):
                is_abort = config["abort_on_next_iteration"]
                hps.in_train_manifest["abort_on_next_iteration"] = is_abort
                if is_abort:
                    logger.warning("Abort flagged in training!")
            if ("chg_eval_interval" in config):
              if (config["chg_eval_interval"] != -1):
                  new_eval_intv = config["chg_eval_interval"]
                  hps.in_train_manifest["chg_eval_interval"] = new_eval_intv
                  hps.train.eval_interval = new_eval_intv
            if (config["best_model"]["losses"] is not None):
              hps.in_train_manifest["best_model"]["losses"] = _deserialize_tensor(config["best_model"]["losses"])
        logger.info(f"Loaded in_train_manifest:\n\t{hps.in_train_manifest}\n")


def save_in_train_manifest(hps, iteration, glb_step, after_save=True):
    hps.in_train_manifest["last_updated"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    hps.in_train_manifest["after_save"] = after_save  # in case of IO exception the manifest is updated before/after
    if (after_save):
      hps.in_train_manifest["iteration"] = iteration
      hps.in_train_manifest["previous_step"] = hps.in_train_manifest["latest_step"] 
      hps.in_train_manifest["latest_step"] = glb_step
    best_losses = hps.in_train_manifest["best_model"]["losses"]
    if (best_losses is not None):
      hps.in_train_manifest["best_model"]["losses"] = [_serialize_tensor(loss) for loss in best_losses]
    try:
      with open(hps.in_train_manifest_path, "w") as f:
        json.dump(hps.in_train_manifest, f, indent=2)
    finally:
      hps.in_train_manifest["best_model"]["losses"] = best_losses
    if (hps.model_sync_folder is not None):
      shutil.copy2(hps.in_train_manifest_path, hps.model_sync_folder)


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warning("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warning("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
