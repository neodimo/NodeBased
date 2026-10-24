"""After a 5 s rest: how long does a GPU->CPU copy of N MB take, repeatedly, with and without compute/copy traffic first?"""
import sys, time, wgpu
mode = sys.argv[1]            # copy33 | copy8 | compute | none
ad = [a for a in wgpu.gpu.enumerate_adapters_sync() if 'discrete' in str(a.info.get('adapter_type','')).lower()][0]
dev = ad.request_device_sync()
def mk(n): return dev.create_buffer(size=n, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
def mkread(n): return dev.create_buffer(size=n, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
big = mk(33*2**20); rb = mkread(33*2**20); rb8 = mkread(8*2**20)
def copy(n, r):
    t = time.perf_counter(); e = dev.create_command_encoder(); e.copy_buffer_to_buffer(big, 0, r, 0, n); dev.queue.submit([e.finish()]); r.map_sync(wgpu.MapMode.READ); r.unmap(); return (time.perf_counter()-t)*1000
for _ in range(3): copy(33*2**20, rb)
time.sleep(6)
out = []
t0 = time.perf_counter()
while time.perf_counter()-t0 < 1.5:
    s = time.perf_counter()
    if mode == 'copy33': out.append((round((s-t0)*1000), round(copy(33*2**20, rb), 1)))
    elif mode == 'copy8': out.append((round((s-t0)*1000), round(copy(8*2**20, rb8), 1)))
    elif mode == 'none': out.append((round((s-t0)*1000), round(copy(33*2**20, rb), 1))) ; time.sleep(0.03)
    time.sleep(max(0, 0.0417-(time.perf_counter()-s)))
print(mode, out)
