import torch
import torch.nn as nn
import torch.onnx

# Define red mínima con una sola convolución
class MinimalConvModel(nn.Module):
    def __init__(self):
        super(MinimalConvModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)

# Instancia y entrada dummy
model = MinimalConvModel()
dummy_input = torch.randn(1, 3, 32, 32)

# Exportar a ONNX
torch.onnx.export(
    model,
    dummy_input,
    "modelo.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Modelo 'modelo.onnx' exportado.")
