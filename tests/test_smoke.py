"""Fast checks that run on CPU in well under a minute."""
import numpy as np
import pytest
import torch

from rs1_beattheta.binaural import measure_beat
from rs1_beattheta.data import CHUNK, TEXTURES, build_dataset
from rs1_beattheta.diffusion import T_STEPS, alpha_bar, noisy_batch
from rs1_beattheta.export import export_hf
from rs1_beattheta.model import UNet1D

TINY = (16, 32, 32, 48, 64)


def test_dataset_shapes_and_level():
    x, y = build_dataset(2, seed=1)
    assert x.shape == (2 * len(TEXTURES), 1, CHUNK) and y.shape == (2 * len(TEXTURES),)
    rms = x.pow(2).mean(dim=(1, 2)).sqrt()
    assert torch.all(rms > 0.1) and torch.all(rms < 0.25)


def test_diffusion_schedule_and_target():
    ac = alpha_bar(T_STEPS)
    assert ac.shape == (T_STEPS,) and ac[0] > 0.99 and 0 < ac[-1] < 0.01
    x0 = torch.randn(3, 1, 64)
    xt, v = noisy_batch(x0, torch.tensor([0, 500, 999]), ac)
    assert xt.shape == x0.shape and v.shape == x0.shape


def test_model_forward():
    net = UNet1D(widths=TINY, n_classes=5).eval()
    out = net(torch.randn(2, 1, CHUNK), torch.tensor([10, 900]), torch.tensor([0, 5]))
    assert out.shape == (2, 1, CHUNK)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    net = UNet1D(widths=TINY, n_classes=5)
    folder = export_hf(tmp_path_factory.mktemp("model"), net.state_dict(), TINY, training_steps=0)
    model = transformers.AutoModel.from_pretrained(str(folder), trust_remote_code=True).eval()
    return net, model


def test_export_weights_match_training_model(exported):
    net, model = exported
    a, b = net.state_dict(), model.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_generate_and_beat_accuracy(exported):
    _, model = exported
    audio = model.generate("ocean_swell", seconds=12, steps=3, seed=1, beat_hz=6.0, carrier_hz=200.0)
    assert audio.ndim == 2 and audio.shape[1] == 2 and audio.dtype == np.float32
    assert abs(measure_beat(audio, model.sample_rate, 200.0, 6.0) - 6.0) < 0.05
    mono = model.generate("warm_pad", seconds=3, steps=3, seed=1)
    assert mono.ndim == 1
