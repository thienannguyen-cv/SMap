import torch

# FACT 1: y = x - x.detach() has value 0 but d y / d x = +1 (unit forward tap)
x = torch.tensor([3.0], requires_grad=True)
y = x - x.detach()
print("F1 value(x-x.detach()) =", y.item(), " (expect 0)")
y.backward()
print("F1 d/dx (x - x.detach()) =", x.grad.item(), " (expect +1)")

# FACT 2: key_query_grdf = -(kq - kq.detach())  -> gradient -1
x = torch.tensor([3.0], requires_grad=True)
y = -(x - x.detach())
y.backward()
print("F2 d/dx -(x - x.detach()) =", x.grad.item(), " (expect -1)")

# FACT 3: diff = sign(a-q).detach() * (a-q).  d diff/d a = sign(a-q)  (so |a-q| surrogate, grad = sign)
a = torch.tensor([5.0], requires_grad=True)
q = torch.tensor([2.0])
diff = torch.sign(a-q).detach()*(a-q)
print("F3 value =", diff.item(), " (expect +3 = |5-2|)")
diff.backward()
print("F3 d/da [sign(a-q).detach()*(a-q)] =", a.grad.item(), " (expect +1 = sign(a-q))")
a = torch.tensor([-4.0], requires_grad=True)
diff = torch.sign(a-q).detach()*(a-q)
diff.backward()
print("F3b a<q: value=", (torch.sign(torch.tensor(-6.0)).item()*-6.0), " d/da =", a.grad.item(), " (expect -1)")

# FACT 4: KEY one — the rectify "attractive" update applied to weights.
# weights = weights.detach() + weights_grdf, where weights_grdf = (factor).detach()*new_r_mask_grdf, new_r_mask_grdf = pre_mask - pre_mask.detach().
# So d weights / d pre_mask = factor.  If factor>0 => increasing weights increases pre_mask param? Let's confirm sign chain.
pre_mask = torch.tensor([0.3], requires_grad=True)
new_r_mask_grdf = pre_mask - pre_mask.detach()
factor = torch.tensor([0.7])  # detached positive
weights = pre_mask.detach() + factor*new_r_mask_grdf
weights.backward()
print("F4 d weights/d pre_mask =", pre_mask.grad.item(), " (expect +0.7 = factor)")
print("F4 note: a loss L that DECREASES with weights -> dL/dpre_mask = (dL/dw)*factor; factor sign sets correction direction.")
