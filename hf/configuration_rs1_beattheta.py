from transformers import PretrainedConfig

TEXTURES = ["warm_pad", "deep_drone", "ocean_swell", "soft_rain", "crystal_pad"]


class RS1BeatThetaConfig(PretrainedConfig):
    model_type = "rs1_beattheta"

    def __init__(self, widths=(48, 96, 128, 192, 256), n_classes=5, textures=None, sr_model=8000,
                 chunk=16384, t_steps=1000, cdim=128, **kwargs):
        self.widths = list(widths)
        self.n_classes = n_classes
        self.textures = list(textures or TEXTURES)
        self.sr_model = sr_model
        self.chunk = chunk
        self.t_steps = t_steps
        self.cdim = cdim
        super().__init__(**kwargs)
