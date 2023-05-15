import os
import sys
import argparse
import logging
import torch

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def list_model_info(model_path):
  assert os.path.isfile(model_path)
  checkpoint_dict = torch.load(model_path, map_location='cpu')
  iteration, _ = _load_checkpoint_dict_entry(checkpoint_dict, 'iteration', raise_if_not_found=True)
  learning_rate, _ = _load_checkpoint_dict_entry(checkpoint_dict, 'learning_rate', raise_if_not_found=True)
  _, has_opt = _load_checkpoint_dict_entry(checkpoint_dict, 'optimizer', raise_if_not_found=False)
  best_losses, has_bl = _load_checkpoint_dict_entry(checkpoint_dict, 'best_losses', None, raise_if_not_found=False)
  gbl_step,_ = _load_checkpoint_dict_entry(checkpoint_dict, 'gbl_step', 0, raise_if_not_found=False)
  logger.info(f"Loaded '{model_path}' ({gbl_step}/{iteration}) [opt:{has_opt}, bl:{has_bl}]")
  if (has_bl):
    logger.info(f"\tbl: {best_losses}")
    logger.info(f"\tbl_sum: {sum(best_losses).item()}")
  logger.info(f"\tlr: {learning_rate}")
  saved_state_dict = checkpoint_dict['model']
  if "enc_p.emb.weight" in saved_state_dict:
    num_rows_enc_checkpoint = saved_state_dict["enc_p.emb.weight"].shape[0]
    logger.info(f"\tenc_p.emb.weight rows: {num_rows_enc_checkpoint}")
  if "emb_g.weight" in saved_state_dict:
    num_rows_emb_checkpoint = saved_state_dict["emb_g.weight"].shape[0]
    logger.info(f"\temb_g.weight rows: {num_rows_emb_checkpoint}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('-m', '--model_path', type=str, default="./build/G^latest.pth")
    args = parser.parse_args()
    model_path = args.model_path

    list_model_info(model_path)
