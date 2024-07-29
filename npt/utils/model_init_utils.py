import wandb
from torch.cuda.amp import GradScaler
from npt.model.npt import NPTModel
from npt.utils.encode_utils import get_torch_dtype
from npt.utils.train_utils import count_parameters, init_optimizer


def init_model_opt_scaler_from_dataset(dataset, c, device=None):
    return init_model_opt_scaler(
        c, metadata=dataset.metadata, device=device)


def init_model_opt_scaler(c, metadata, device=None):
    if device is None:
        device = c.exp_device

    model = NPTModel(
        c, metadata=metadata, device=device)

    model_torch_dtype = get_torch_dtype(dtype_name=c.model_dtype)
    model = model.to(device=device).type(model_torch_dtype)
    print(f'Model has {count_parameters(model)} parameters,'
          f'batch size {c.exp_batch_size}.')

    optimizer = init_optimizer(
        c=c, model_parameters=model.parameters(), device=device)
    print(f'Initialized "{c.exp_optimizer}" optimizer.')

    # Automatic Mixed Precision (AMP)
    # If c.model_amp is False, the GradScaler call becomes a no-op
    # so we can switch between default/mixed precision without if/else
    # statements.
    scaler = GradScaler(enabled=c.model_amp)
    if c.model_amp:
        print(f'Initialized gradient scaler for Automatic Mixed Precision.')

    return model, optimizer, scaler
