import torch

a = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
idx = torch.LongTensor([[0, 1]]).view(-1, 1)
print(a.gather(dim=-1, index=idx))

# python3 mello_scripts/tool/gather_test.py