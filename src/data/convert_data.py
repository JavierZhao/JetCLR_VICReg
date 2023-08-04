import torch


def convert_x(bb, device):
    """
    Converts Batch.x to a tensor of shape (batch_size, 3, n_constit),
    where n_constit is the maximum number of constituents in the batch.
    zeros are padded to the right.
    """
    bb.x = bb.x[:, :3]
    bb.x = bb.x[:, [2, 0, 1]]
    # Compute number of constituents for each item in the batch
    n_constits = torch.bincount(bb.batch)

    # Compute maximum number of constituents
    n_constit = n_constits.max().item()

    # Allocate a tensor of the desired shape, filled with a padding value (e.g. zero)
    x_padded = torch.zeros(bb.batch.max().item() + 1, 3, n_constit)

    # Fill the padded tensor with the values from Batch.x
    for i, (start, length) in enumerate(zip(bb.ptr[:-1], n_constits)):
        x_padded[i, :, :length] = bb.x[start : start + length].t()

    return x_padded.to(device)
