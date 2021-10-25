import torch
import numpy as np



lol = torch.rand(size = (1,), dtype = torch.float32)
lol.cuda()
print(lol.device);