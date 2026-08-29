import torch

if __name__ == "__main__":
    x = torch.rand(3, 4, 5)
    print(x, x.shape)
    (b, s, d) = x.shape
    print(b, s, d)
