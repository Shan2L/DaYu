import os
import sys
import torch
sys.path.append('../../')

from climax import ModelConfigGlobal, ClimaX
from DaYu.asyncPipelineModel import AsyncPipelineModel

# 2. Model setup
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

# 3. Device setup and model transfer
device = torch.device("cuda:0")
model = model.to(device)
model.eval()

# 4. Input preparation
batch = 30
x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).to(device)
lead_times = torch.tensor([72]*batch, dtype=torch.float32).to(device)
variables = model_config.default_vars
out_variables = model_config.out_variables

# 5. Model inference
inputs = (x, None, lead_times, variables, out_variables, None, None)

degree = 10
asyn_model = AsyncPipelineModel(model, degree)
x_ = torch.chunk(x, degree, 0)
lead_times_ = torch.chunk(lead_times, degree, 0)
print(lead_times_)

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

with torch.no_grad():
    golden = model(*inputs)

    golden = torch.chunk(golden, 10, 0)
    for (i, result) in enumerate(asyn_model._forward(sliced_inputs)):
        print(torch.allclose(golden[i], result, 1e-6, 1e-6))

