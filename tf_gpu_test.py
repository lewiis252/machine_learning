import torch

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Using device {device}.")


print(torch.rand(2,3).cuda())

print(torch.cuda.is_available())