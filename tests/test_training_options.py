"""Schedule, initialisation and optimizer options of the deletion task. None of them changes the model class."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from editflows import DataBatch, LinearKappaSchedule
from editflows.deletion_task import OPTIM_DEFAULTS, initialize
from editflows.edit_head import EncoderWithEditHead
from editflows.encoder import LlamaEncoder
from editflows.encoder_pretraining import MaskedLetterPretraining
from editflows.lr_scheduler import WarmUpCosineAnnealing, WarmUpCosineGroups
from editflows.schedules import ExponentialKappaSchedule

ROOT = Path(__file__).resolve().parents[1]
VOCAB, BOS = 21, 20

try:
    OmegaConf.register_new_resolver("eval", eval)  # train.py registers the same resolver
except ValueError:
    pass


def make_model(seed=0):
    torch.manual_seed(seed)
    encoder = LlamaEncoder(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, max_seq_len=64)
    return EncoderWithEditHead(encoder=encoder, vocab_size=VOCAB)


def batch():
    return DataBatch.from_sequences([torch.tensor([BOS, 1, 2, 3, 4, 5]), torch.tensor([BOS, 7, 8, 9])])


# ---- kappa schedule ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rate,normalize", [(6.0, True), (6.0, False), (0.5, True), (-3.0, True)])
def test_exponential_schedule_is_a_valid_kappa(rate, normalize):
    kappa = ExponentialKappaSchedule(rate, normalize)
    t = torch.linspace(0, 1, 1001, dtype=torch.float64)
    k = kappa(t)
    assert k[0] == 0 and (k.diff() > 0).all()
    assert math.isclose(float(k[-1]), 1.0 if normalize else 1 - math.exp(-rate), rel_tol=1e-12)

    # loss_coefficient is kappa' / (1 - kappa) wherever the clamp of 1 - kappa is not active
    inner = t[(1 - k) > 2e-3][1:-1]
    eps = 1e-6
    hazard = (kappa(inner + eps) - kappa(inner - eps)) / (2 * eps) / (1 - kappa(inner))
    assert torch.allclose(kappa.loss_coefficient(inner), hazard, rtol=1e-5)


@pytest.mark.parametrize("normalize", [True, False])
def test_growing_hazard_schedule_is_a_valid_kappa(normalize):
    kappa = ExponentialKappaSchedule(3.0, normalize, growth=1.5)
    t = torch.linspace(0, 1, 1001, dtype=torch.float64)
    k = kappa(t)
    assert k[0] == 0 and (k.diff() > 0).all()
    total = 3.0 * math.expm1(1.5) / 1.5  # the cumulative hazard at t = 1
    assert math.isclose(float(k[-1]), 1.0 if normalize else 1 - math.exp(-total), rel_tol=1e-12)

    inner = t[(1 - k) > 2e-3][1:-1]
    eps = 1e-6
    hazard = (kappa(inner + eps) - kappa(inner - eps)) / (2 * eps) / (1 - kappa(inner))
    assert torch.allclose(kappa.loss_coefficient(inner), hazard, rtol=1e-5)
    if not normalize:
        assert torch.allclose(kappa.loss_coefficient(t), 3.0 * torch.exp(1.5 * t))


def test_exponential_schedule_without_normalisation_has_a_constant_hazard():
    coefficient = ExponentialKappaSchedule(6.0, normalize=False).loss_coefficient(torch.linspace(0, 1, 11))
    assert torch.equal(coefficient, torch.full((11,), 6.0))


def test_normalised_hazard_stays_within_a_small_factor_of_the_rate():
    coefficient = ExponentialKappaSchedule(6.0).loss_coefficient(torch.linspace(0, 1, 1001, dtype=torch.float64))
    assert coefficient.min() >= 6.0 and coefficient.max() < 6.0 * 4


def test_small_rate_approaches_the_linear_schedule():
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    assert torch.allclose(ExponentialKappaSchedule(1e-5)(t), LinearKappaSchedule()(t), atol=1e-5)


# ---- initialisation ---------------------------------------------------------------------------------


def test_head_initialisation_sets_the_rates_and_keeps_the_model():
    model = make_model()
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    initialize(model, head_log_rates=[-9.95, -9.95, -1.1])
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes

    x_t = batch()
    rate = model(x_t, torch.rand(x_t.batch_size))
    expected = torch.tensor([-9.95, -9.95, -1.1])[:, None].expand(3, len(x_t.x))
    assert torch.allclose(rate.log_lambdas, expected, atol=1e-4)

    # The last layer still receives a gradient that depends on the token, so the rates can leave the start
    rate.log_lambda_del[1].backward()
    assert model.head.lambda_head[-1].weight.grad.abs().sum() > 0


def test_time_embedding_can_start_at_zero():
    model = make_model().eval()
    initialize(model, time_embedding_zero=True)
    x_t = batch()
    with torch.no_grad():
        early, late = model(x_t, torch.zeros(2)), model(x_t, torch.ones(2))
    assert torch.equal(early.log_lambdas, late.log_lambdas)


# ---- optimizer parameter groups ---------------------------------------------------------------------


def schedule(make, steps, groups=1):
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(groups)]
    opt = torch.optim.SGD([{"params": [p], "lr": 3e-4 * 0.5**i} for i, p in enumerate(params)], lr=3e-4)
    scheduler = make(opt)
    lrs = []
    for _ in range(steps):
        lrs.append([g["lr"] for g in opt.param_groups])
        opt.step()
        scheduler.step()
    return np.array(lrs)


def test_grouped_schedule_equals_the_plain_one_for_a_single_group():
    kwargs = dict(warmup_steps=10, max_steps=50, start_lr=1e-8, end_lr=3e-6)
    plain = schedule(lambda opt: WarmUpCosineAnnealing(opt, **kwargs), 50)
    grouped = schedule(lambda opt: WarmUpCosineGroups(opt, **kwargs), 50)
    assert np.allclose(plain, grouped, rtol=1e-9, atol=0)


def test_grouped_schedule_delays_a_group():
    kwargs = dict(warmup_steps=2, max_steps=40, start_lr=1e-8, end_lr=3e-6)
    lrs = schedule(lambda opt: WarmUpCosineGroups(opt, **kwargs, delays=[0, 5], ramps=[0, 4]), 12, groups=2)
    assert (lrs[:5, 1] == 0).all() and (lrs[:, 0] > 0).all()
    # After the delay the second group ramps up to its own base learning rate (half of the first)
    assert np.allclose(lrs[5:9, 1] / lrs[5:9, 0], 0.5 * np.array([0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(lrs[9:, 1] / lrs[9:, 0], 0.5)


def test_task_builds_parameter_groups():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="train", overrides=[
            "experiment=deletion-smoke", "name=test", "task.optim.decay_matrices_only=true",
            "task.optim.encoder_lr_scale=0.1", "task.optim.encoder_freeze_steps=5", "task.optim.encoder_ramp_steps=5",
        ])
    OmegaConf.resolve(config)
    task = instantiate(config.task, seed_sequence=np.random.SeedSequence(0))
    task.model = make_model()
    task.lr_scheduler_config.max_steps = 100

    configured = task._grouped_optimizers()
    opt, scheduler = configured["optimizer"], configured["lr_scheduler"]["scheduler"]
    grouped = [p for g in opt.param_groups for p in g["params"]]
    assert len(grouped) == len(list(task.model.parameters())) == len({id(p) for p in grouped})

    encoder = {id(p) for p in task.model.encoder.parameters()}
    lr = config.task.optimizer.lr
    for group, delay in zip(opt.param_groups, scheduler.delays):
        in_encoder = {id(p) in encoder for p in group["params"]}
        assert len(in_encoder) == 1
        assert math.isclose(group["initial_lr"], lr * (0.1 if in_encoder.pop() else 1.0))
        assert delay == (5 if id(group["params"][0]) in encoder else 0)
        if group["weight_decay"] == 0:
            assert all(p.ndim < 2 for p in group["params"])
        else:
            assert all(p.ndim >= 2 for p in group["params"])
    assert opt.param_groups[0]["weight_decay"] > 0 and id(opt.param_groups[0]["params"][0]) not in encoder


def test_default_options_keep_the_plain_optimizer():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="train", overrides=["experiment=deletion-smoke", "name=test"])
    OmegaConf.resolve(config)
    task = instantiate(config.task, seed_sequence=np.random.SeedSequence(0))
    assert task.init_config == {}
    assert task.optim_config == OPTIM_DEFAULTS


def test_scaled_trunk_initialisation_shrinks_the_residual_branches():
    model = make_model()
    layers = model.encoder.model.layers
    before = [layer.self_attn.o_proj.weight.std().item() for layer in layers]
    untouched = layers[0].self_attn.q_proj.weight.clone()
    initialize(model, trunk="scaled")
    expected = 0.02 / math.sqrt(2 * len(layers))
    for layer, old in zip(layers, before):
        assert abs(layer.self_attn.o_proj.weight.std().item() - expected) < 0.2 * expected < old
        assert abs(layer.mlp.down_proj.weight.std().item() - expected) < 0.2 * expected
    assert torch.equal(layers[0].self_attn.q_proj.weight, untouched)


def test_encoder_starts_from_a_pretraining_checkpoint(tmp_path):
    pretrained = make_model(seed=1).encoder
    # as saved by the masked-letter pretraining: the encoder under "encoder.", plus the discarded decoder
    state = {f"encoder.{k}": v for k, v in pretrained.state_dict().items()}
    state["decoder.weight"] = torch.zeros(20, 32)
    torch.save({"state_dict": state}, tmp_path / "encoder.ckpt")

    model = make_model(seed=2)
    head_before = model.head.lambda_head[0].weight.clone()
    initialize(model, encoder_checkpoint=str(tmp_path / "encoder.ckpt"))
    for (name, a), b in zip(model.encoder.state_dict().items(), pretrained.state_dict().values()):
        assert torch.equal(a, b), name
    assert torch.equal(model.head.lambda_head[0].weight, head_before)


def test_model_starts_from_the_evaluated_weights_of_another_run(tmp_path):
    raw, averaged = make_model(seed=1), make_model(seed=3)
    # as saved by the task: the raw model under "model.", the EMA copy under "ema_model.ema_model."
    state = {f"model.{k}": v for k, v in raw.state_dict().items()}
    state |= {f"ema_model.ema_model.{k}": v for k, v in averaged.state_dict().items()}
    torch.save({"state_dict": state}, tmp_path / "run.ckpt")

    model = make_model(seed=2)
    initialize(model, checkpoint=str(tmp_path / "run.ckpt"))
    for (name, a), b in zip(model.state_dict().items(), averaged.state_dict().values()):
        assert torch.equal(a, b), name

    # without an EMA copy the raw weights are used
    torch.save({"state_dict": {k: v for k, v in state.items() if k.startswith("model.")}}, tmp_path / "raw.ckpt")
    initialize(model, checkpoint=str(tmp_path / "raw.ckpt"))
    for (name, a), b in zip(model.state_dict().items(), raw.state_dict().values()):
        assert torch.equal(a, b), name


def test_masked_letter_pretraining_step():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="train", overrides=["task=mlm", "experiment=deletion-smoke", "name=test"])
    OmegaConf.resolve(config)
    task = instantiate(config.task, seed_sequence=np.random.SeedSequence(0))
    assert isinstance(task, MaskedLetterPretraining)
    task.configure_model()
    x_1 = DataBatch.from_sequences([torch.cat((torch.tensor([BOS]), torch.randint(0, 20, (n,)))) for n in (60, 80)])
    loss, accuracy = task._step(x_1, torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and 0 <= accuracy <= 1
    # an untrained encoder predicts 20 letters at chance: the loss is about log(20)
    assert abs(loss.item() - math.log(20)) < 0.5
    loss.backward()
    assert task.decoder.weight.grad.abs().sum() > 0
    # the state dict carries the encoder under "encoder.", which is what the deletion task loads
    assert any(k.startswith("encoder.model.layers.0") for k in task.state_dict())


def test_head_bias_can_be_set_while_the_weights_stay():
    model = make_model()
    weights = model.head.lambda_head[-1].weight.clone()
    initialize(model, head_log_rates=[-9.95, -9.95, -1.1], head_weight_scale=1.0)
    assert torch.equal(model.head.lambda_head[-1].weight, weights)
    x_t = batch()
    rate = model(x_t, torch.rand(x_t.batch_size))
    # the rates scatter around the chosen ones instead of sitting exactly on them
    assert (rate.log_lambda_ins < -9).all() and rate.log_lambda_del.std() > 0
    assert abs(rate.log_lambda_del.mean().item() + 1.1) < 2.0
