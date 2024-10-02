import torch
import torch_tensorrt
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            bias = torch.FloatTensor([0.0, 1.0, 10.0])
            weight = torch.FloatTensor([-1.0, 0.0, 1.0]).reshape(3, 1, 1, 1)
            self.conv.bias.copy_(bias) 
            self.conv.weight.copy_(weight)

    def forward(self, x):
        return self.conv(self.act(x))

def generate_ts():
    # This will be either done in hematoscope-dev before building the image,
    # or perhaps locally cloned during the image generation. Either way, we
    # probably don't want any nn.Module definitions in hematoscope-inference.
    model = Model().eval().cuda()
    script_module = torch.jit.script(model)
    script_module.save("ir.ts")
    print("TS model built")

def load_ts():
    model = torch.jit.load("ir.ts")
    print("TS model loaded")
    return model

def run_model():
    inputs = torch.ones((1, 1, 12, 12)).cuda()
    model = torch_tensorrt.load("trt.ts").cuda()
    res = model(inputs).cpu()
    target = torch.as_tensor([-1.0, 1.0, 11.0]).reshape(3, 1, 1)
    assert torch.allclose(res, target)
    print("Model ran successfully")

if __name__ == "__main__":
    generate_ts() 

    # Actual entry point for the build script
    script_module = load_ts()
    inputs = [torch.randn((1, 1, 12, 12)).cuda()]
    trt_gm = torch_tensorrt.compile(script_module, ir="torchscript", inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)
    print("Compiled tensorrt engine")

    # Entry for running inference
    run_model() 
