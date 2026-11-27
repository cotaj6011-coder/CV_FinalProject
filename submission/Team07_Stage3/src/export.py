"""Export the selected EMA checkpoint for inference without altering its training snapshot."""
import hashlib
import json

import torch
from ultralytics.utils.torch_utils import strip_optimizer

from model import ROOT, read_config


def export_inference_checkpoint(source, destination):
    snapshot_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = torch.load(source, map_location="cpu", weights_only=False)
    original_model = snapshot["ema"] if snapshot.get("ema") is not None else snapshot["model"]
    original_state = original_model.state_dict()
    strip_optimizer(f=source, s=str(destination))
    exported = torch.load(destination, map_location="cpu", weights_only=False)
    exported_state = exported["model"].state_dict()
    equal = original_state.keys() == exported_state.keys() and all(
        original_state[key].dtype == exported_state[key].dtype
        and torch.equal(original_state[key], exported_state[key]) for key in original_state)
    assert equal, "Inference export changed the selected EMA state tensors."
    assert hashlib.sha256(source.read_bytes()).hexdigest() == snapshot_hash
    evidence = {
        "source_checkpoint": source.relative_to(ROOT).as_posix(),
        "inference_checkpoint": destination.relative_to(ROOT).as_posix(),
        "training_snapshot_bytes": source.stat().st_size,
        "inference_checkpoint_bytes": destination.stat().st_size,
        "training_snapshot_sha256": snapshot_hash,
        "inference_checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "export_method": "Ultralytics strip_optimizer; selected EMA preserved; original snapshot retained",
        "stored_model_precision": "FP16 floating-point tensors; integer buffers unchanged",
        "evaluation_precision": "FP32 (half=False)",
        "state_tensor_count": len(original_state),
        "model_tensors_equal": equal,
        "validation_equal": None,
        "validation_note": "Not yet independently re-evaluated after this export.",
    }
    (ROOT / "experiments" / "checkpoint_export.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    selection = json.loads((ROOT / "experiments" / "selection.json").read_text(encoding="utf-8"))
    print(json.dumps(export_inference_checkpoint(
        ROOT / selection["source_checkpoint"], ROOT / read_config()["checkpoint_file"]), indent=2))
