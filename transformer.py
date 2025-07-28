import torch


# mathematical trick for incremental averages using matrix multiplication
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b

print(f"a={a}")
print("--")
print(f"b={b}")
print("--")
print(f"c={c}")


# vectorized bag of words
