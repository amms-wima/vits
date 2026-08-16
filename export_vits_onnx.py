import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Try importing standard VITS components
try:
    import utils
    from models import SynthesizerTrn
except ImportError:
    print(
        "Error: Could not import 'utils' or 'SynthesizerTrn'. "
        "Run this script from the root of your VITS repository."
    )
    sys.exit(1)

import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic


def remove_all_weight_norm(model: nn.Module):
    """Recursively removes weight normalization from all submodules in PyTorch."""
    for m in model.modules():
        # Legacy torch.nn.utils.remove_weight_norm
        try:
            torch.nn.utils.remove_weight_norm(m)
        except (ValueError, AttributeError):
            pass

        # PyTorch 2.x parametrizations.weight_norm
        try:
            if hasattr(
                torch.nn.utils, "parametrize"
            ) and torch.nn.utils.parametrize.is_parametrized(m, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
        except Exception:
            pass


class PiperVITSWrapper(nn.Module):
    """Wrapper module that exposes the standard Piper ONNX interface."""

    def __init__(self, vits_model, is_multispeaker=False):
        super().__init__()
        self.vits_model = vits_model
        self.is_multispeaker = is_multispeaker

    def forward(self, input, input_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        if self.is_multispeaker and sid is not None:
            audio = self.vits_model.infer(
                input,
                input_lengths,
                sid=sid,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                max_len=None,
            )[0]
        else:
            audio = self.vits_model.infer(
                input,
                input_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                max_len=None,
            )[0]

        return audio


def export_fp32_onnx(
    config_path: str,
    checkpoint_path: str,
    output_fp32_path: str,
    opset_version: int = 15,
):
    """Loads PyTorch VITS weights and exports a full-precision FP32 ONNX graph."""
    print(f"Loading config from: {config_path}")
    hps = utils.get_hparams_from_file(config_path)

    is_multispeaker = hasattr(hps.data, "n_speakers") and hps.data.n_speakers > 1
    num_speakers = getattr(hps.data, "n_speakers", 0)

    print(
        f"Building SynthesizerTrn (Multi-speaker: {is_multispeaker}, Speakers: {num_speakers})..."
    )
    net_g = SynthesizerTrn(
        len(hps.symbols)
        if hasattr(hps, "symbols")
        else hps.data.filter_length // 2 + 1,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=num_speakers if is_multispeaker else 0,
        **hps.model,
    )

    _ = net_g.eval()

    print(f"Loading checkpoint: {checkpoint_path}")
    utils.load_checkpoint(checkpoint_path, net_g, None)

    # Safely remove weight normalization across all child modules
    remove_all_weight_norm(net_g)

    wrapper = PiperVITSWrapper(net_g, is_multispeaker=is_multispeaker)
    wrapper.eval()

    # Dummy inputs for tracing
    dummy_input = torch.randint(low=1, high=10, size=(1, 15), dtype=torch.int64)
    dummy_input_lengths = torch.tensor([15], dtype=torch.int64)
    dummy_scales = torch.tensor([0.333, 1.0, 0.5], dtype=torch.float32)

    input_names = ["input", "input_lengths", "scales"]
    dynamic_axes = {
        "input": {0: "batch", 1: "phonemes"},
        "input_lengths": {0: "batch"},
        "scales": {0: "num_scales"},
        "output": {0: "batch", 2: "audio_samples"},
    }

    dummy_args = [dummy_input, dummy_input_lengths, dummy_scales]

    if is_multispeaker:
        dummy_sid = torch.tensor([0], dtype=torch.int64)
        dummy_args.append(dummy_sid)
        input_names.append("sid")
        dynamic_axes["sid"] = {0: "batch"}

    print(f"Exporting FP32 ONNX model to: {output_fp32_path}...")
    torch.onnx.export(
        wrapper,
        tuple(dummy_args),
        output_fp32_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print("FP32 Export Complete.")


def create_quantized_onnx(fp32_path: str, quant_path: str):
    """Creates a dynamic INT8 quantized version optimized for low-latency mobile/browser runtimes."""
    print(f"Quantizing {fp32_path} -> {quant_path}...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=quant_path,
        weight_type=QuantType.QUInt8,  # Fixed: QUInt8 instead of QU8
    )
    print("Quantization Complete.")

def generate_piper_json_sidecar(
    config_path: str, output_json_path: str, is_multispeaker: bool = True
):
    """Generates the required Piper .onnx.json metadata file."""
    with open(config_path, "r", encoding="utf-8") as f:
        vits_config = json.load(f)

    piper_config = {
        "audio": {
            "sample_rate": vits_config.get("data", {}).get(
                "sampling_rate", 22050
            )
        },
        "espeak": {"voice": "en-us"},
        "inference": {
            "noise_scale": 0.333,
            "length_scale": 1.0,
            "noise_w": 0.5,
        },
        "phoneme_type": "ipa",
        "phoneme_map": {},
        "phoneme_id_map": {},
        "num_symbols": len(vits_config.get("symbols", [])),
        "num_speakers": vits_config.get("data", {}).get("n_speakers", 1),
        "speaker_id_map": {},
    }

    if "symbols" in vits_config:
        piper_config["phoneme_id_map"] = {
            s: [i] for i, s in enumerate(vits_config["symbols"])
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(piper_config, f, indent=2)

    print(f"Created Piper metadata file: {output_json_path}")


if __name__ == "__main__":
    CONFIG_JSON = "/home/ash/prj/pretrained_models/ft_ashinw_combo129-455000/config.json"
    CHECKPOINT_PTH = "/home/ash/prj/pretrained_models/ft_ashinw_combo129-455000/G^latest.pth"

    OUTPUT_FP32_ONNX = "/home/ash/prj/pretrained_models/ft_ashinw_combo129-455000/pali_vits_fp32.onnx"
    OUTPUT_QUANT_ONNX = "/home/ash/prj/pretrained_models/ft_ashinw_combo129-455000/pali_vits_quant.onnx"
    OUTPUT_PIPER_JSON = "/home/ash/prj/pretrained_models/ft_ashinw_combo129-455000/pali_vits_fp32.onnx.json"

    export_fp32_onnx(
        config_path=CONFIG_JSON,
        checkpoint_path=CHECKPOINT_PTH,
        output_fp32_path=OUTPUT_FP32_ONNX,
        opset_version=15,
    )

    create_quantized_onnx(
        fp32_path=OUTPUT_FP32_ONNX, quant_path=OUTPUT_QUANT_ONNX
    )

    generate_piper_json_sidecar(
        config_path=CONFIG_JSON, output_json_path=OUTPUT_PIPER_JSON
    )