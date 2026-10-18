"""Research-only retained texture surface probe; run from repo root with xcb Qt."""
import argparse, json, os, time, statistics, collections
import numpy as np
import PySide6
from PySide6 import QtCore, QtWidgets
import wgpu, rendercanvas
from rendercanvas.qt import QRenderWidget
p=argparse.ArgumentParser(); p.add_argument('--adapter',default='integrated'); p.add_argument('--out',required=True); p.add_argument('--capture',action='store_true'); p.add_argument('--frames',type=int,default=300); a=p.parse_args()
app=QtWidgets.QApplication([])
canvas=QRenderWidget(size=(1920,1080), present_method='screen', update_mode='manual')
canvas.show(); app.processEvents()
context=canvas.get_wgpu_context()
adapters=wgpu.gpu.enumerate_adapters_sync()
adapter=next(x for x in adapters if a.adapter.lower() in str(x.info.get('adapter_type','')).lower())
device=adapter.request_device_sync()
fmt=context.get_preferred_format(adapter); context.configure(device=device,format=fmt,alpha_mode='opaque')
counts=collections.Counter()
def count(cls,name):
    orig=getattr(cls,name)
    def wrapped(*args,**kwargs): counts[name]+=1; return orig(*args,**kwargs)
    setattr(cls,name,wrapped)
for cls,names in [(type(device.queue),['read_texture','read_buffer','write_texture','write_buffer']), (wgpu.backends.wgpu_native.GPUCommandEncoder,['copy_texture_to_buffer','copy_buffer_to_texture'])]:
    for name in names: count(cls,name)
tex=device.create_texture(size=(1920,1080,1),format='rgba8unorm-srgb',usage=wgpu.TextureUsage.TEXTURE_BINDING|wgpu.TextureUsage.COPY_DST)
y,x=np.indices((1080,1920)); pixels=np.empty((1080,1920,4),np.uint8); pixels[:,:,:3]=np.where(((x//64+y//64)%2)[...,None],180,50); pixels[:,:,3]=255
device.queue.write_texture({'texture':tex},pixels,{'bytes_per_row':1920*4},(1920,1080,1))
shader=device.create_shader_module(code='''
@group(0) @binding(0) var img: texture_2d<f32>;
struct Out { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->Out {
 var p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
 var o:Out; o.p=vec4f(p[i],0.,1.); o.uv=p[i]*vec2f(0.5,-0.5)+vec2f(0.5); return o;
}
@fragment fn fs(o:Out)->@location(0) vec4f { return textureLoad(img,vec2i(clamp(o.uv,vec2f(0.),vec2f(.9999))*vec2f(1920.,1080.)),0); }
''')
pipeline=device.create_render_pipeline(layout='auto',vertex={'module':shader,'entry_point':'vs'},fragment={'module':shader,'entry_point':'fs','targets':[{'format':fmt}]})
bind=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':tex.create_view()}])
# Overlay uses a scissored second GPU pass; no Qt bitmap overlay or texture upload.
over=device.create_shader_module(code='''
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {var p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.,1.);}
@fragment fn fs()->@location(0) vec4f {return vec4f(1.,.05,.02,1.);}
''')
op=device.create_render_pipeline(layout='auto',vertex={'module':over,'entry_point':'vs'},fragment={'module':over,'entry_point':'fs','targets':[{'format':fmt}]})
frame=[0]; stamps=[]; drawms=[]; acquirems=[]; presentms=[]; sizes=[]; rss=[]
def memory(): return int(open('/proc/self/statm').read().split()[1])*os.sysconf('SC_PAGE_SIZE')/1048576
def draw():
    t=time.perf_counter(); view=context.get_current_texture().create_view(); acquirems.append((time.perf_counter()-t)*1000)
    enc=device.create_command_encoder(); rp=enc.begin_render_pass(color_attachments=[{'view':view,'resolve_target':None,'clear_value':(0,0,0,1),'load_op':'clear','store_op':'store'}])
    rp.set_pipeline(pipeline); rp.set_bind_group(0,bind); rp.draw(3)
    w,h=canvas.get_physical_size(); rp.set_pipeline(op); rp.set_scissor_rect((frame[0]*7)%max(1,w-80),h//2,80,12); rp.draw(3); rp.end(); device.queue.submit([enc.finish()]); drawms.append((time.perf_counter()-t)*1000)
    sizes.append([w,h]); frame[0]+=1
orig_present=context._rc_present
def present(**kwargs):
    t=time.perf_counter(); result=orig_present(**kwargs); presentms.append((time.perf_counter()-t)*1000); stamps.append(time.perf_counter()); return result
context._rc_present=present
canvas.request_draw(draw)
# Warm pipelines and acquisition, then collect 300 complete application present calls.
for _ in range(20): canvas.force_draw(); app.processEvents()
frame[0]=0; drawms.clear(); acquirems.clear(); presentms.clear(); stamps.clear(); sizes.clear(); counts.clear()
t0=time.perf_counter(); cpu0=time.process_time()
resized=False
while len(stamps)<a.frames:
    i=len(stamps)
    if i>=150 and not resized:
        canvas.resize(1600,900); resized=True
    app.processEvents()
    if len(stamps)<a.frames: canvas.force_draw()
    if i%25==0: rss.append([i,memory()])
elapsed=time.perf_counter()-t0; cpu=time.process_time()-cpu0
assert len(stamps)==a.frames,(len(stamps),a.frames)
assert counts['read_texture']==counts['read_buffer']==counts['write_texture']==counts['copy_texture_to_buffer']==counts['copy_buffer_to_texture']==0,dict(counts)
def dist(v): return {k:float(np.percentile(v,q)) for k,q in [('p50',50),('p95',95),('p99',99),('max',100)]}
out={'versions':{'PySide6':PySide6.__version__,'Qt':QtCore.qVersion(),'wgpu':wgpu.__version__,'rendercanvas':rendercanvas.__version__},'adapter':dict(adapter.info),'all_adapters':[dict(x.info) for x in adapters],'platform':app.platformName(),'dpr':canvas.devicePixelRatioF(),'screen_refresh_hz':app.primaryScreen().refreshRate(),'context':type(context).__name__,'format':fmt,'frames':len(stamps),'elapsed_s':elapsed,'cpu_percent_one_core':100*cpu/elapsed,'acquire_ms':dist(acquirems),'draw_submit_ms':dist(drawms),'present_call_ms':dist(presentms),'present_return_interval_ms':dist(np.diff(stamps)*1000),'sizes':list(map(list,sorted(set(map(tuple,sizes))))),'rss_mib':rss,'transfer_calls_after_warmup':dict(counts),'initial_upload_bytes':pixels.nbytes,'bounds':'CPU timestamps around draw/submit and native present return; no scanout or photon timing; RSS is process resident memory, not VRAM','draw_ms_raw':drawms,'present_ms_raw':presentms,'present_return_stamps':stamps}
open(a.out,'w').write(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if not k.endswith('_raw') and k!='present_return_stamps'},indent=2))
# Deliberate validation capture AFTER all timings and transfer counters are serialized.
if a.capture:
    import subprocess
    canvas.raise_(); canvas.activateWindow()
    for _ in range(20): app.processEvents(); time.sleep(.01)
    subprocess.run(['spectacle','-b','-n','-a','-e','-S','-o',os.path.abspath(a.out+'.png')],check=True,timeout=15)
context.unconfigure(); canvas.close()
