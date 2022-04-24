import torch


print("Device capability:", torch.cuda.get_device_capability())
print("num devices:", torch.cuda.device_count())

print("testing torch:")
print("version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

a = torch.cuda.FloatTensor(2).zero_()
print("Tensor a =", str(a))
b = torch.randn(2).cuda()
print("Tensor b =", str(b))
c = a + b
print("Tensor c =", str(c))
