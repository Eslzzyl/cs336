import numpy.typing as npt
import numpy as np
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    input_batch_list = []
    target_batch_list = []
    for _ in range(batch_size):
        max_start_index = len(dataset) - context_length
        index = np.random.randint(0, max_start_index)
        input_seq = torch.Tensor(dataset[index : index + context_length], device=device)
        target_seq = torch.Tensor(
            dataset[index + 1 : index + context_length + 1], device=device
        )
        input_batch_list.append(input_seq)
        target_batch_list.append(target_seq)
    input_batch = torch.vstack(input_batch_list)
    target_batch = torch.vstack(target_batch_list)
    return (input_batch, target_batch)
