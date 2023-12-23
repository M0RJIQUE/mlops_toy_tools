import torch


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

model = torch.load("../checkpoints/dense.pt").eval().to("cpu")
traced_model = torch.jit.trace(model, torch.randn(1, 64).to("cpu"))
torch.jit.save(traced_model, "pt-dense/1/model.pt")
