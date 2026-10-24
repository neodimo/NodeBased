"""After a 6 s rest: per-tick copy GPU->CPU of `mb` MB at 24 fps for 1.5 s; prints copy time per tick. Optional extra dummy copy of `xmb` MB per tick."""
import sys, time, wgpu
mb = float(sys.argv[1]); xmb = float(sys.argv[2]) if len(sys.argv) > 2 else 0
ad = [a for a in wgpu.gpu.enumerate_adapters_sync() if 'discrete' in str(a.info.get('adapter_type','')).lower()][0]
dev = ad.request_device_sync()
big = dev.create_buffer(size=40*2**20, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
rb = dev.create_buffer(size=40*2**20, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
def copy(n):
    t = time.perf_counter(); e = dev.create_command_encoder(); e.copy_buffer_to_buffer(big, 0, rb, 0, int(n*2**20)//4*4); dev.queue.submit([e.finish()]); rb.map_sync(wgpu.MapMode.READ); rb.unmap(); return (time.perf_counter()-t)*1000
for _ in range(3): copy(33)
time.sleep(6)
out = []; t0 = time.perf_counter()
while time.perf_counter()-t0 < 1.6:
    s = time.perf_counter()
    a = copy(mb)
    if xmb: copy(xmb)
    out.append((round((s-t0)*1000), round(a, 1)))
    time.sleep(max(0, 0.0417-(time.perf_counter()-s)))
print(mb, xmb, out)
