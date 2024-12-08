import sys
sys.path.append('../../')
from simpleModel import SimpleModel
from DaYu.asyncPipelineModel import AsyncPipelineModel
from torch.profiler import profile, ProfilerActivity
import torch


model = SimpleModel(256, 256).cuda()
model.eval()
x = torch.randn(100, 4096, 256).float().cuda()
for name, _ in model.named_modules():
    print(name)


async_model = AsyncPipelineModel(model, 4)
async_model.register_dependency("layer12", 0, "layer1", 1)
async_model.register_dependency("layer10", 1, "layer3", 2)
async_model.register_dependency("layer9", 2, "layer4", 3)


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
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./dayu_native')) as prof:
    for iter in range(15):
        print(iter)
        for output in async_model(x):
            print(output.shape)
        prof.step()



