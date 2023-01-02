import torch
aa = torch.randn((5, 2))
labels = torch.Tensor([1, 0, 0, 0, 1])
print(aa)
print(labels)

mask = (labels == 0)
print(mask)

print(aa[mask])
center = aa[mask].mean(dim=0)
print(center)

print(center.tolist())

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/tool/test.py
