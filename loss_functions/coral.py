import torch

# def CORAL(source, target):
#     d = source.size(1)
#     ns, nt = source.size(0), target.size(0)

#     # source covariance
#     tmp_s = torch.ones((1, ns)).cuda() @ source
#     cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

#     # target covariance
#     tmp_t = torch.ones((1, nt)).cuda() @ target
#     ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

#     # frobenius norm
#     loss = (cs - ct).pow(2).sum()
#     loss = loss / (4 * d * d)

#     return loss

def CORAL(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    print(xm,xc,xmt,xct)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)
    return loss

# Source: https://ssarcandy.tw/2017/10/31/deep-coral/
