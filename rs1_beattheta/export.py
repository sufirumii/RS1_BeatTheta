"""Step 4 - Export: turn trained weights into a folder that loads with transformers AutoModel."""
import json
import shutil
from pathlib import Path

from safetensors.torch import save_file

from .data import SR, CHUNK, TEXTURES
from .diffusion import T_STEPS

HF_DIR = Path(__file__).resolve().parent.parent / "hf"


def export_hf(out_dir, state_dict, widths, training_steps):
    """Write weights, config, model code and model card into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()}, str(out / "model.safetensors"))
    config = {
        "model_type": "rs1_beattheta",
        "architectures": ["RS1BeatTheta"],
        "auto_map": {
            "AutoConfig": "configuration_rs1_beattheta.RS1BeatThetaConfig",
            "AutoModel": "modeling_rs1_beattheta.RS1BeatTheta",
        },
        "widths": list(widths),
        "n_classes": len(TEXTURES),
        "textures": TEXTURES,
        "sr_model": SR,
        "chunk": CHUNK,
        "t_steps": T_STEPS,
        "cdim": 128,
        "training_steps": training_steps,
        "torch_dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name in ("configuration_rs1_beattheta.py", "modeling_rs1_beattheta.py"):
        shutil.copy(HF_DIR / name, out / name)
    shutil.copy(HF_DIR / "MODEL_CARD.md", out / "README.md")
    return out
