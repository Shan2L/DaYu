import os
import sys
import torch
from simpleModel import SimpleModel 
from torch.profiler import profile, ProfilerActivity
from threading import Thread


model = SimpleModel(256, 256).cuda()
model.eval()
model = torch.compile(model)

streams = [torch.cuda.Stream() for _ in range(8)]
x1 = torch.randn(512, 256).cuda()
x2 = torch.randn(512, 256).cuda()
x3 = torch.randn(512, 256).cuda()
x4 = torch.randn(512, 256).cuda()
x5 = torch.randn(512, 256).cuda()
x6 = torch.randn(512, 256).cuda()
x7 = torch.randn(512, 256).cuda()
x8 = torch.randn(512, 256).cuda()

def run_model(data, stream):
    with torch.no_grad():
        with torch.cuda.stream(stream):
            output = model(data)
            print(output.shape)






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
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./mt')) as prof:
    with torch.no_grad():
        for iter in range(15):
            print(iter)
            thread1 = Thread(target=run_model, args=(x1, streams[0]))
            thread2 = Thread(target=run_model, args=(x2, streams[1]))
            thread3 = Thread(target=run_model, args=(x3, streams[2]))
            thread4 = Thread(target=run_model, args=(x4, streams[3]))
            thread5 = Thread(target=run_model, args=(x5, streams[4]))
            thread6 = Thread(target=run_model, args=(x6, streams[5]))
            thread7 = Thread(target=run_model, args=(x7, streams[6]))
            thread8 = Thread(target=run_model, args=(x8, streams[7]))
            thread1.start()
            thread2.start()
            thread3.start()
            thread4.start()
            thread5.start()
            thread6.start()
            thread7.start()
            thread8.start()
            prof.step()
