import utils
import torch
import argparse


device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('-m', '--model_path', type=str, default="./build/G_latest.pth")
    args = parser.parse_args()

    model_path = args.model_path
    _ = utils.query_checkpoint(model_path)
