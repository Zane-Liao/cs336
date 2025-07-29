import torch
import numpy as np
import numpy.typing as npt

# Solution
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((dataset[i:i+context_length]).astype(np.int64)) for i in starting_idxs])
    y = torch.stack([torch.from_numpy((dataset[i+1:i+1+context_length]).astype(np.int64)) for i in starting_idxs])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def load(src, model, optimizer):
    check_point = torch.load(src, map_location=torch.device('cpu'))
    
    model.load_state_dict(check_point['model_state_dict'])
    
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    
    iteration = check_point['iteration']
    
    print(f"load {src} iterations: {iteration}")
    
    # model.train()
    return iteration


def save(model, optimizer, iteration, out):
    check_point = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(check_point, out) 
    print(f"save {out} iterations: {iteration}")