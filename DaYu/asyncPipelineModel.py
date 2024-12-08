import torch
import torch.nn as nn

class AsyncPipelineModel():
    def __init__(self, model:nn.Module, degree=2):
        self.model = model
        self.degree = degree
        self.streams = [torch.cuda.Stream() for _ in range(self.degree)]
    
    def _slice_input(self, params):
        if type(params) == torch.Tensor:
            return torch.chunk(params, self.degree, 0)
        elif type(params) == tuple:
            processed_params = []
            for param in params:
                if param is None:
                    processed_params.append(param)
                elif param is torch.Tensor:
                    processed_params.append(torch.chunk(params, self.degree, 0))
                else:
                    raise ValueError("The type is not supported now.")

            sliced_params = [[]] * self.degree
            for pparams in processed_params:
                for i in range(self.degree):
                    sliced_params[i].append(pparams[i])
            return tuple(sliced_params)
                
        else:
            raise ValueError("The type is not supported now.")
        
    def register_dependency(self, mod_name1: str, stream_id1: int, mod_name2: str, stream_id2: int):
        module1 = None
        module2 = None
        for name, mod in self.model.named_modules():
            if name == mod_name1:
                if module2 is None:
                    raise ValueError("Module2 must be in front of module1")
                print("[INFO] find module1: name")
                module1 = mod
            if name == mod_name2:
                print("[INFO] find module2: name")
                module2 = mod

        if module1 is not None and module2 is not None:
            event = torch.cuda.Event()
            def forward_hook(module, input, output):
                event.record(self.streams[stream_id1])
            def pre_forward_hook(module, input):
                self.streams[stream_id2].wait_event(event)
            module1.register_forward_hook(forward_hook)
            module2.register_forward_pre_hook(pre_forward_hook)
        else:
            raise ValueError("One of module not found in the modle.")

    def _forward(self, x: tuple):
        for i in range(self.degree):
            with torch.cuda.stream(self.streams[i]):
                output = self.model(*x[i])
                yield output      
        

    def __call__(self, x):
        sliced_params = self._slice_input(x)
        for i in range(self.degree):
            with torch.cuda.stream(self.streams[i]):
                output = self.model(sliced_params[i])
                yield output



