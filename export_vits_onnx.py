import os
import json
import torch
import torch.nn as nn

# Load your custom VITS definitions
from models import SynthesizerTrn
from text.symbols import symbols

def export_onnx(config_path, checkpoint_path, output_onnx_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # 1. Instantiate SynthesizerTrn matching your 2023 config
    net_g = SynthesizerTrn(
        len(symbols),
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        n_speakers=config["data"]["n_speakers"],
        **config["model"]
    )

    # 2. Load PyTorch Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        net_g.load_state_dict(checkpoint["model"])
    else:
        net_g.load_state_dict(checkpoint)

    # Ensure model is set to evaluation mode
    net_g.eval()

    n_speakers = config["data"]["n_speakers"]

    # 3. Define Deterministic VITS Generator Wrapper
    class VITSGeneratorONNX(nn.Module):
        def __init__(self, generator, num_speakers):
            super().__init__()
            self.generator = generator
            self.num_speakers = num_speakers

        def forward(self, x, x_lengths, scales, sid=None):
            # Extract noise & length parameters
            noise_scale = scales[0]
            length_scale = scales[1]
            noise_scale_w = scales[2]

            # Handle speaker ID embedding tensor requirement
            if self.num_speakers > 0 and sid is None:
                sid = torch.tensor([0], dtype=torch.long)

            with torch.no_grad():
                audio = self.generator.infer(
                    x,
                    x_lengths,
                    sid=sid,
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                    noise_scale_w=noise_scale_w
                )[0]
            return audio

    onnx_wrapper = VITSGeneratorONNX(net_g, n_speakers)
    onnx_wrapper.eval()

    # Dummy input tensors
    dummy_x = torch.randint(0, len(symbols), (1, 30), dtype=torch.long)
    dummy_x_lengths = torch.tensor([30], dtype=torch.long)
    dummy_scales = torch.tensor([0.0, 1.0, 0.8], dtype=torch.float32) # noise_scale = 0.0
    dummy_sid = torch.tensor([0], dtype=torch.long) # Default speaker index 0

    print("Exporting ONNX graph (using TorchScript legacy tracer)...")
    
    # Force legacy exporter (dynamo=False)
    torch.onnx.export(
        onnx_wrapper,
        (dummy_x, dummy_x_lengths, dummy_scales, dummy_sid),
        output_onnx_path,
        export_params=True,
        opset_version=15,  # opset 14 or 15 works best for VITS ops
        do_constant_folding=True,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phoneme_sequence"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "audio_samples"}
        },
        dynamo=False  # <--- CRITICAL: Disables torch.export / Dynamo tracer
    )

    # 4. Save metadata config JSON for client runtime
    # Ensure mandatory control symbols exist in mapping
    phoneme_id_map = {symbol: [i] for i, symbol in enumerate(symbols)}

    # Fallback/Safety mapping for Piper's default BOS (^), EOS ($), and PAD ( ) tokens
    # If they are already in symbols.symbols, dictionary update will retain their true indices.
    if "^" not in phoneme_id_map and "_" in phoneme_id_map:
        phoneme_id_map["^"] = phoneme_id_map["_"] # Map BOS to padding index if missing
    if "$" not in phoneme_id_map and "_" in phoneme_id_map:
        phoneme_id_map["$"] = phoneme_id_map["_"] # Map EOS to padding index if missing
    if " " not in phoneme_id_map and "_" in phoneme_id_map:
        phoneme_id_map[" "] = phoneme_id_map["_"]

    piper_config = {
        "audio": {
            "sample_rate": config["data"]["sampling_rate"]
        },
        "espeak": {
            "voice": "en-us"
        },
        "phoneme_type": "text", # or "espeak" depending on your symbol set
        "phoneme_map": {},
        "phoneme_id_map": phoneme_id_map,
        "num_symbols": len(symbols),
        "num_speakers": n_speakers
    }

    config_output = output_onnx_path + ".json"
    with open(config_output, "w") as f:
        json.dump(piper_config, f, indent=2)

    print(f"Success! Model saved to {output_onnx_path} and {config_output}")

if __name__ == "__main__":
    export_onnx("models/config.json", "models/G^latest.pth", "models/pali_vits.onnx")