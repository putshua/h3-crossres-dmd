import copy
import pytest
import torch
from h3dmd.ltx_upsampling import LTXLatentUpsampler
from h3dmd.upsampling import mapper_spec,new_mapper,reconstruction_losses
from h3dmd.upsampler_training import load_settings


CONFIG='configs/h3_upsampler_ltx128_full998k_local4_pilot_20261013.json'


def test_ltx_128_architecture_and_artifact_spec():
    settings,cfg=load_settings(CONFIG)
    m=new_mapper(cfg)
    assert sum(p.numel() for p in m.parameters())==8578584
    spec=mapper_spec(cfg)
    assert spec['temporal_mixing'] is True and spec['blocks_per_stage']==4
    assert spec['normalization']=='h3_standardized_video_v1'
    cfg.crossres.mapper_gradient_checkpointing=True
    assert mapper_spec(cfg)==spec  # Training memory policy is not an architecture change.
    cfg.crossres.mapper_kind='rcan'
    assert mapper_spec(cfg)==dict(kind='rcan',channels=24,width=128,groups=4,
        low_pixels=[320,512],high_pixels=[480,768],spatial_compression=16,fps=24,
        temporal_mixing=False,normalization='h3_standardized_video_v1')


def test_ltx_geometry_temporal_context_and_training_update():
    torch.set_num_threads(2);torch.manual_seed(42)
    m=LTXLatentUpsampler((6,6),width=32,blocks=1)
    x=torch.randn(1,24,3,4,4,requires_grad=True)
    y=m(x);assert y.shape==(1,24,3,6,6)
    altered=x.detach().clone();altered[:,:,1]+=2
    assert not torch.allclose(y[:,:,0],m(altered)[:,:,0])
    before=m.tail.weight.detach().clone()
    opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
    loss,_=reconstruction_losses(y,torch.randn_like(y));loss.backward();opt.step()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert not torch.equal(before,m.tail.weight)
    for p in m.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    with pytest.raises(ValueError,match='exact 1.5x'):
        m(torch.randn(1,24,3,5,4))


def test_ltx_checkpointing_preserves_values_and_gradients():
    torch.set_num_threads(2);torch.manual_seed(42)
    a=LTXLatentUpsampler((6,6),width=32,blocks=1)
    b=copy.deepcopy(a);b.gradient_checkpointing=True
    x=torch.randn(1,24,3,4,4)
    ya,yb=a(x),b(x)
    torch.testing.assert_close(ya,yb,rtol=0,atol=0)
    ya.square().mean().backward();yb.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
    b.load_state_dict(a.state_dict(),strict=True)
