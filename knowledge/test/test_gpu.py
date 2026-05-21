import torch
#查看当前GPU是否可用
print("GPU是否可用")
print(torch.cuda.is_available())
#查看GPU数量
print("GPU数量")
print(torch.cuda.device_count())
#查看GPU名字
print("GPU名字")
print(torch.cuda.get_device_name(0))
