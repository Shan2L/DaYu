import os
import sys
import torch
from simpleModel import SimpleModel 
from torch.profiler import profile, ProfilerActivity


model = SimpleModel(256, 256).cuda()
model.eval()
model = torch.compile(model)
x = torch.randn(4096, 256).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=torch.profiler.schedule(
                                                wait=1,
                                                warmup=5,
                                                active=3,
                                                repeat=2),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./baseline')) as prof:
    with torch.no_grad():
        for iter in range(15):
            print(iter)
            golden = model(x)
            prof.step()

