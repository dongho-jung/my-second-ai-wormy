# The question is training, not inference: forward + backward + Adam on one
# minibatch, which is what a PPO update actually spends its time on.
import time, torch, torch.nn as nn

class Conv(nn.Module):
    def __init__(s, ch=32, hid=256):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(3, ch // 2, 3, 2, 1), nn.ReLU(),
                               nn.Conv2d(ch // 2, ch, 3, 2, 1), nn.ReLU(), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(ch * 64 + 128, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 16))
    def forward(s, p, v): return s.head(torch.cat([s.conv(p), v], 1))

class Mlp(nn.Module):
    # no conv: terrain as a few hundred raycast/occupancy floats in the vector
    def __init__(s, obs=384, hid=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(obs, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 16))
    def forward(s, p, v): return s.net(v)

def step_bench(make, dev, batch, threads, iters=30, obs=384, conv=True):
    torch.set_num_threads(threads)
    m = make().to(dev)
    opt = torch.optim.Adam(m.parameters(), 3e-4)
    p = torch.randn(batch, 3, 32, 32, device=dev) if conv else torch.zeros(1)
    v = torch.randn(batch, 128 if conv else obs, device=dev)
    tgt = torch.randn(batch, 16, device=dev)
    for _ in range(5):
        opt.zero_grad(); ((m(p, v) - tgt) ** 2).mean().backward(); opt.step()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(); ((m(p, v) - tgt) ** 2).mean().backward(); opt.step()
    if dev == "mps": torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return batch / dt

print("=== conv policy (0.63M params), one PPO minibatch = fwd + bwd + Adam ===")
for dev, threads in [("cpu", 1), ("cpu", 10), ("mps", 10)]:
    r = step_bench(Conv, dev, 1024, threads)
    print(f"  {dev} threads={threads:2}: {r:10,.0f} samples/s  -> 10M steps in {10e6/r/60:6.1f} min of compute")
print()
print("=== vector-only policy, no conv (0.17M params) ===")
for dev, threads in [("cpu", 1), ("cpu", 10), ("mps", 10)]:
    r = step_bench(Mlp, dev, 1024, threads, conv=False)
    print(f"  {dev} threads={threads:2}: {r:10,.0f} samples/s  -> 10M steps in {10e6/r/60:6.1f} min of compute")
