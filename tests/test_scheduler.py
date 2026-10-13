import torch

from h3dmd.h3.scheduler import H3DualNoiseScheduler, shifted_sigma


def test_shared_q_produces_native_dual_schedule() -> None:
    scheduler = H3DualNoiseScheduler(video_shift=12, audio_shift=3)
    level = scheduler.level(0.5)
    assert torch.isclose(level.video_sigma, torch.tensor(12 / 13))
    assert torch.isclose(level.audio_sigma, torch.tensor(3 / 4))
    assert torch.equal(level.video_timestep, 1 - level.video_sigma)
    assert torch.equal(level.audio_timestep, 1 - level.audio_sigma)

    schedule = scheduler.schedule(50)
    assert schedule.num_forwards == 49
    assert schedule.q[0] == 1 and schedule.q[-1] == 0
    assert schedule.video_sigmas[0] == 1 and schedule.video_sigmas[-1] == 0
    assert schedule.audio_sigmas[0] == 1 and schedule.audio_sigmas[-1] == 0


def test_dataward_velocity_and_plus_sign_x0_reconstruction() -> None:
    scheduler = H3DualNoiseScheduler()
    x0 = torch.randn(2, 3, 4)
    epsilon = torch.randn_like(x0)
    sigma = shifted_sigma(torch.tensor(0.37), 12)
    xt = scheduler.add_noise(x0, epsilon, sigma)
    velocity = scheduler.velocity_target(x0, epsilon)
    assert torch.allclose(velocity, x0 - epsilon)
    assert torch.allclose(scheduler.to_x0(xt, velocity, sigma), x0, atol=1e-6)

    sigma_next = torch.tensor(0.23)
    stepped = scheduler.euler_step(xt, velocity, sigma, sigma_next)
    expected = (1 - sigma_next) * x0 + sigma_next * epsilon
    assert torch.allclose(stepped, expected, atol=1e-6)


def test_explicit_few_step_grid_is_q_not_modality_sigma() -> None:
    scheduler = H3DualNoiseScheduler()
    schedule = scheduler.schedule(q=[1.0, 0.7, 0.2, 0.0])
    assert schedule.num_forwards == 3
    assert torch.allclose(schedule.video_sigmas, shifted_sigma(schedule.q, 12))
    assert torch.allclose(schedule.audio_sigmas, shifted_sigma(schedule.q, 3))
