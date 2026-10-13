"""One runtime precision policy for training and standalone H3 inference."""
import torch


def configure_h3_precision(config):
    enabled = config.training.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
