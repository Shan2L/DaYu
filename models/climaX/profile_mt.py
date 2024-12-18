import os
import sys
import torch
sys.path.append('../../')
from climax import ModelConfigGlobal, ClimaX
from DaYu.asyncPipelineModel import AsyncPipelineModel
from torch.profiler import profile, ProfilerActivity
from threading import Thread, Event
import time
import random

class MyThread(Thread):
    def __init__(self, id):
        super().__init__()

    def run(self):
        with torch.cuda.stream(streams[i]):
            output = model(*sliced_inputs[i])

# register slot for given operator in target stream
def register_dependency(mod_name: str, stream_id: int, time_slot: int):

    def pre_forward_hook(module, input):
        # print("pre_forward_hook registerd")
        print(f"{torch.cuda.current_stream() == streams[stream_id]}")
        if torch.cuda.current_stream() == streams[stream_id]:
            print(f"sleep for {time_slot} s")
            time.sleep(time_slot)

    for name, mod in model.named_modules():
        if name == mod_name:
            # print(f"[INFO] find module2: {name}")
            mod.register_forward_pre_hook(pre_forward_hook)

# Model setup
model_config = ModelConfigGlobal()
model = ClimaX(
    default_vars=model_config.default_vars,
    img_size=model_config.img_size,
    patch_size=model_config.patch_size,
    embed_dim=model_config.embed_dim,
    depth=model_config.depth,
    decoder_depth=model_config.decoder_depth,
    num_heads=model_config.num_heads,
    mlp_ratio=model_config.mlp_ratio,
    drop_path=model_config.drop_path,
    drop_rate=model_config.drop_rate,
)


# Device setup and model transfer
device = torch.device("cuda:0")
model = model.to(device)
model.eval()

# Input preparation
batch = 30
x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).to(device)
lead_times = torch.tensor([72]*batch, dtype=torch.float32).to(device)
variables = model_config.default_vars
out_variables = model_config.out_variables

# Model inference
inputs = (x, None, lead_times, variables, out_variables, None, None)
del(inputs)

degree = 10
asyn_model = AsyncPipelineModel(model, degree)
x_ = torch.chunk(x, degree, 0)
lead_times_ = torch.chunk(lead_times, degree, 0)

sliced_inputs = []
for i in range(degree):
    params = []
    params.append(x_[i])
    params.append(None)
    params.append(lead_times_[i])
    params.append(variables)
    params.append(out_variables)
    params.append(None)
    params.append(None)
    sliced_inputs.append(tuple(params))

sliced_inputs = tuple(sliced_inputs)

streams = [torch.cuda.Stream() for _ in range(degree)]

threads = [ MyThread(i) for i in range(degree)]


# generate random time slots for each operator
ENABLE_OPERATOR_SLOT = False
ENABLE_LEADING_SLOT = True

slot_dict = {}
for degree_id in range(degree):
    degree_slot_dict = {}

    degree_slot_dict["leading_slot"] = round(random.uniform(0, 0.1), 4) if ENABLE_LEADING_SLOT else 0
            
    for name, mod in model.named_modules():
        time_slot = 0
        if ENABLE_OPERATOR_SLOT:
            if random.randint(0, 1000) % 100 != 0:
                time_slot = 0
            else:
                time_slot = round(random.uniform(0, 0.01), 4)
        degree_slot_dict[name] = time_slot
    slot_dict[degree_id] = degree_slot_dict

# if operator time slot is enabled, used generated slot to register pre_forward_hook for corresponding module
if ENABLE_OPERATOR_SLOT:
    for i in range(degree):
        for name, mod in model.named_modules():
            register_dependency(name, i, slot_dict[i][name])

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./multi_thread')) as prof:
    with torch.no_grad():
        for i in range(degree):
            threads[i].start()
            if ENABLE_LEADING_SLOT:
                time.sleep(slot_dict[i]["leading_slot"])
        for i in range(degree):
            threads[i].join()
        print(f"max cuda memory usage: {torch.cuda.max_memory_allocated()/ 1024 / 1024/1024}")
