# What one decision costs on this machine, and what the same net costs batched
# during training. Single sample = the live game path; batch = the trainer.
import time, torch, torch.nn as nn
torch.set_num_threads(1)  # the live driver gets one core, not ten

class Policy(nn.Module):
    def __init__(self, ch=32, hid=256, patch=32, state=128, actions=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, ch // 2, 3, 2, 1), nn.ReLU(),     # 32 -> 16
            nn.Conv2d(ch // 2, ch, 3, 2, 1), nn.ReLU(),    # 16 -> 8
            nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(ch * (patch // 4) ** 2 + state, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, actions))
    def forward(self, patch, state):
        return self.head(torch.cat([self.conv(patch), state], 1))

def bench(dev, batch, iters, ch=32, hid=256):
    m = Policy(ch, hid).to(dev).eval()
    p, s = torch.randn(batch, 3, 32, 32, device=dev), torch.randn(batch, 128, device=dev)
    with torch.inference_mode():
        for _ in range(20): m(p, s)
        if dev == "mps": torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): m(p, s)
        if dev == "mps": torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / iters
    n = sum(x.numel() for x in m.parameters())
    return dt, n, batch / dt

print("torch", torch.__version__, "| mps available:", torch.backends.mps.is_available())
for ch, hid, label in [(32, 256, "small  "), (64, 512, "medium "), (128, 1024, "large  ")]:
    dt, n, _ = bench("cpu", 1, 300, ch, hid)
    print(f"{label} {n/1e6:5.2f} M params ({n*4/1048576:5.1f} MB) | 1 core, batch 1: {dt*1000:6.3f} ms/decision -> {1/dt:7.0f} decisions/s | 60 Hz needs {dt*60*100:5.2f}% of one core")
print()
for dev in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
    for batch in [1, 64, 1024]:
        dt, n, thru = bench(dev, batch, 100 if batch > 1 else 300)
        print(f"small net on {dev:3} batch {batch:4}: {dt*1000:7.3f} ms -> {thru:10,.0f} samples/s")
