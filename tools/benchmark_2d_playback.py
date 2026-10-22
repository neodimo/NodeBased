#!/usr/bin/env python3
"""Real-window uncached/cached playback benchmark for the Lane 8 compositing path.

Generated EXR inputs and run JSON belong in the caller-supplied scratch/output directory;
only the small .nbcomp graph and this reproducible generator are tracked.
"""
from __future__ import annotations
import argparse, cProfile, json, os, pathlib, statistics, subprocess, sys, threading, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/playback_comp.nbcomp'


def make_inputs(directory: pathlib.Path, width: int, height: int, frames: int = 48):
    import numpy as np
    from nodebased.media import write_exr
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base_x = xx / max(1, width - 1)
    base_y = yy / max(1, height - 1)
    for role, phase in (('plate', 0), ('background', 11)):
        for frame in range(1, frames + 1):
            shift = (frame * 19 + phase * 31) % width
            band = ((xx + shift) % max(1, width // 12) < max(1, width // 24)).astype(np.float32)
            pixels = np.empty((height, width, 4), np.float32)
            pixels[..., 0] = 0.08 + 0.34 * base_x + 0.08 * band
            pixels[..., 1] = 0.07 + 0.28 * base_y + 0.12 * ((frame + phase) % 7) / 7
            pixels[..., 2] = 0.10 + 0.24 * (1.0 - base_x) + 0.06 * band
            pixels[..., 3] = 1.0
            # A moving soft-edged oval makes sequence frames visibly distinct without noisy
            # pixels that would turn the fixture into a compression benchmark.
            cx = (frame * 23 + phase * 17) % width
            cy = height * (0.32 + 0.28 * ((frame + phase) % 13) / 13)
            rad = ((xx - cx) / max(1, width * .11)) ** 2 + ((yy - cy) / max(1, height * .13)) ** 2
            mask = np.clip(1.0 - rad, 0.0, 1.0)
            pixels[..., 0] += mask * 0.32
            pixels[..., 1] += mask * 0.10
            write_exr(directory / f'{role}.{frame:04d}.exr', pixels, bits='half')
    return {'width': width, 'height': height, 'frames': frames,
            'plate_pattern': str(directory / 'plate.%04d.exr'),
            'background_pattern': str(directory / 'background.%04d.exr')}


def percentile(values, q):
    return round(float(__import__('numpy').percentile(values, q)), 3) if values else None


def run_window(args):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    sys.path.insert(0, str(ROOT))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    import numpy as np
    import nodebased.gpu3d as gpu3d
    if args.force_adapter != 'default':
        original_state = gpu3d._state
        gpu3d._state = lambda choice=None: original_state(args.force_adapter if (choice or 'default') == 'default' else choice)
    from nodebased.app import Window, STYLE
    import nodebased.app as app_module
    from nodebased.core import load_document
    from nodebased import media, imaging
    from nodebased.playback import DisplayCache
    from nodebased.tileexec import TileRegion

    events=[]; shown=[]; statuses=[]; worker_profiles=[]; sample_counts={'qt':{},'worker':{}}; node_costs={}
    lock=threading.Lock(); start=time.perf_counter(); run_end=[None]
    def mark(stage, **kw):
        with lock: events.append({'stage':stage,'t_ms':round((time.perf_counter()-start)*1000,3),**kw})
    def sample_stacks():
        while run_end[0] is None:
            with lock: frames=sys._current_frames()
            for thread in threading.enumerate():
                if thread.ident not in frames: continue
                lane='qt' if thread is threading.main_thread() else ('worker' if thread.name.startswith('nodebased-preview') else None)
                if lane is None: continue
                frame=frames[thread.ident]
                key=f'{pathlib.Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}'
                sample_counts[lane][key]=sample_counts[lane].get(key,0)+1
            time.sleep(.002)
    doc=load_document(args.comp)
    doc['nodes']['plate']['params']['path']=str(pathlib.Path(args.plate_pattern).resolve())
    doc['nodes']['background']['params']['path']=str(pathlib.Path(args.background_pattern).resolve())
    args.width,args.height=((1920,1080) if args.size=='1080p' else (3840,2160))
    sx=args.width/1920; sy=args.height/1080
    doc['nodes']['roto']['params'].update(width=args.width,height=args.height)
    for shape in doc['node_data']['roto']['shapes']:
        for point in shape['points']:
            for field, scale in (('x',sx),('y',sy),('in_x',sx),('out_x',sx),('in_y',sy),('out_y',sy)):
                scalar=point[field]
                if isinstance(scalar,dict):
                    scalar['value']*=scale
                    for key in scalar.get('curve',{}).get('keys',[]): key['value']*=scale
                else: point[field]=scalar*scale
    app=QApplication.instance() or QApplication([]); app.setQuitOnLastWindowClosed(False); app.setApplicationName('NodeBased'); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    win=Window(doc); win.show(); app.processEvents()
    win.proxy.setCurrentIndex(win.proxy.findData(1))
    win.show_thumbnails=False
    # Keep the cached run's scene-linear frames warm before the measured playback interval.
    if args.cache_mode=='cached':
        target=doc['view']; ident=DisplayCache.identity(doc,target,1)
        for frame in range(1,49):
            region=win.tile_executor.canvas_region(doc,target,frame=frame,tier=1)
            result=win.tile_executor.compose_region(doc,target,region,frame=frame,tier=1)
            key=DisplayCache.key(doc,target,frame,1)
            win.display_cache.put(key,result.pixels,(region.x,region.y,region.width,region.height),lazy=True)
        mark('cache_prewarm_complete',frames=48,display_cache_bytes=win.display_cache.bytes)
    # Stage boundaries; evaluation node timings are appended from Evaluator.profile_log.
    original_tick=win.playback_tick
    def playback_tick(*a,**kw): mark('frame_tick',frame=int(win.dispatcher.document['time']['current'])); return original_tick(*a,**kw)
    win.playback_tick=playback_tick
    original_request=win.request_preview
    def request(*a,**kw): mark('frame_request',frame=int(win.dispatcher.document['time']['current'])); return original_request(*a,**kw)
    win.request_preview=request
    original_tile=win.tile_executor._render_tile
    from nodebased.tiles import SUPPORTED_TILED_KINDS
    def render_tile(document,node_id,frame,tier,region,node_digests,cancel):
        t=time.perf_counter(); out=original_tile(document,node_id,frame,tier,region,node_digests,cancel)
        kind=document['nodes'][node_id]['type']; key=(int(frame),node_id)
        row=node_costs.setdefault(key,{'frame':int(frame),'node':node_id,'kind':kind,'path':'tile' if kind in SUPPORTED_TILED_KINDS else 'full-frame fallback','tile_evaluations':0,'elapsed_ms':0.0})
        row['tile_evaluations']+=1; row['elapsed_ms']+=(time.perf_counter()-t)*1000
        return out
    win.tile_executor._render_tile=render_tile
    original_compose=win.tile_executor.compose_region
    def compose(*a,**kw):
        t=time.perf_counter(); mark('compose_start',frame=kw.get('frame'),target=a[1]); before={k:len(v) for k,v in win.evaluator.profile_log.items()}
        result=original_compose(*a,**kw)
        fallback=list(result.fallback_nodes)
        mark('compose_end',elapsed_ms=round((time.perf_counter()-t)*1000,3),tiled=result.tiled,
             fallback_nodes=fallback,tile_hits=result.tile_hits,tile_misses=result.tile_misses)
        for node,rows in win.evaluator.profile_log.items():
            for row in rows[before.get(node,0):]: mark('full_frame_node_eval',frame=row['frame'],node=node,elapsed_ms=row['wall_time_ms'],cache_hits=row['cache_hits'],path='full-frame fallback')
        mark('node_evaluation_summary',frame=kw.get('frame'),path='tile' if result.tiled else 'full-frame fallback',fallback_nodes=fallback,
             nodes=[{'node':r['node'],'path':r['path'],'tile_evaluations':r['tile_evaluations'],'elapsed_ms':round(r['elapsed_ms'],3)} for r in node_costs.values() if r['frame']==int(kw.get('frame') or 0)])
        return result
    win.tile_executor.compose_region=compose
    original_put=win.display_cache.put
    def put(*a,**kw):
        t=time.perf_counter(); result=original_put(*a,**kw); mark('cache_put',elapsed_ms=round((time.perf_counter()-t)*1000,3),bytes=win.display_cache.bytes); return result
    win.display_cache.put=put
    original_qimage=app_module.to_qimage
    def qimage(*a,**kw):
        t=time.perf_counter(); image=original_qimage(*a,**kw); mark('display_conversion',elapsed_ms=round((time.perf_counter()-t)*1000,3)); return image
    app_module.to_qimage=qimage
    original_read=imaging.read_image
    def read(*a,**kw):
        t=time.perf_counter(); out=original_read(*a,**kw); mark('plate_read_decode',elapsed_ms=round((time.perf_counter()-t)*1000,3),path=str(a[0] if a else kw.get('path',''))); return out
    imaging.read_image=read
    original_media_read=media.read_media_raster
    def read_media(*a,**kw):
        t=time.perf_counter(); out=original_media_read(*a,**kw); mark('plate_read_decode',elapsed_ms=round((time.perf_counter()-t)*1000,3),path=str(a[0] if a else '')); return out
    media.read_media_raster=read_media
    original_ready=Window.preview_ready
    def ready(self,payload,frame,image,status,render_region=None):
        req,cancel=payload
        mark('ui_handoff',frame=req.frame,status=status.split('  ·  ')[0],has_frame=frame is not None)
        if frame is not None:
            shown.append((time.perf_counter(),req.frame)); statuses.append(status.split('  ·  ')[0])
        original_ready(self,payload,frame,image,status,render_region)
    Window.preview_ready=ready
    original_submit=win.executor.submit
    job_counter=[0]
    def submit(fn,*a,**kw):
        job_counter[0]+=1; job_id=job_counter[0]; job_name=getattr(fn,'__name__','worker')
        def wrapped(*wa,**wk):
            mark('worker_start',job_id=job_id,job=job_name,thread=threading.current_thread().name); prof=cProfile.Profile(); prof.enable()
            try: return fn(*wa,**wk)
            finally:
                prof.disable(); stream=__import__('io').StringIO(); __import__('pstats').Stats(prof,stream=stream).sort_stats('cumtime').print_stats(50)
                worker_profiles.append(stream.getvalue()); mark('worker_end',job_id=job_id,job=job_name,thread=threading.current_thread().name)
        return original_submit(wrapped,*a,**kw)
    win.executor.submit=submit
    sampler=threading.Thread(target=sample_stacks,name='playback-sampler',daemon=True); sampler.start()
    def begin():
        nonlocal start
        start=time.perf_counter(); mark('playback_start',mode=args.cache_mode,adapter=args.adapter)
        win.toggle_playback(True)
        QTimer.singleShot(round(args.seconds*1000),finish)
    def finish():
        win.toggle_playback(False); win.timer.stop(); win.preview_queue.cancel()
        mark('playback_stop',frames=len(shown))
        def quit_when_idle():
            if win.busy:
                QTimer.singleShot(100,quit_when_idle); return
            win.saved_document=__import__('copy').deepcopy(win.dispatcher.document)
            run_end[0]=time.perf_counter(); app.quit()
        QTimer.singleShot(100,quit_when_idle)
    QTimer.singleShot(300,begin)
    app.exec()
    sampler.join(timeout=.2)
    intervals=[(shown[i][0]-shown[i-1][0])*1000 for i in range(1,len(shown))]
    frame_times=[]
    last=None
    for event in events:
        if event['stage']=='worker_start': last=event['t_ms']
        elif event['stage']=='ui_handoff' and last is not None and event['has_frame']:
            frame_times.append(event['t_ms']-last); last=None
    result={'case':args.case,'resolution':f'{args.width}x{args.height}','duration_s':args.seconds,
        'cache_mode':args.cache_mode,'adapter':args.adapter,'force_adapter':args.force_adapter,
        'frames_displayed':len(shown),'achieved_fps':round(len(shown)/args.seconds,3),
        'frame_interval_ms':{'p50':percentile(intervals,50),'p95':percentile(intervals,95),'p99':percentile(intervals,99)},
        'worker_to_handoff_ms':{'p50':percentile(frame_times,50),'p95':percentile(frame_times,95),'p99':percentile(frame_times,99)},
        'status_line':statuses[-1] if statuses else win.statusBar().currentMessage(),
        'per_node':sorted(node_costs.values(),key=lambda row:row['elapsed_ms'],reverse=True),'timeline':events,
        'sampling_profile':sample_counts,'worker_call_profiles':worker_profiles}
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    win.close()
    print(json.dumps({k:result[k] for k in ('case','resolution','cache_mode','adapter','achieved_fps','frames_displayed','frame_interval_ms','worker_to_handoff_ms','status_line')}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--generate-inputs',type=pathlib.Path)
    p.add_argument('--size',choices=('1080p','4k'),default='1080p')
    p.add_argument('--frames',type=int,default=48)
    p.add_argument('--comp',type=pathlib.Path,default=FIXTURE)
    p.add_argument('--plate-pattern'); p.add_argument('--background-pattern')
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('playback.json'))
    p.add_argument('--seconds',type=float,default=10.0)
    p.add_argument('--cache-mode',choices=('uncached','cached'),default='uncached')
    p.add_argument('--adapter',default='unknown')
    p.add_argument('--force-adapter',choices=('default','integrated','cpu'),default='default')
    p.add_argument('--case',default='1080p')
    p.add_argument('--display',default=os.environ.get('DISPLAY'))
    a=p.parse_args()
    if a.self_check:
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); sys.path.insert(0,str(ROOT))
        from nodebased.core import load_document
        d=load_document(FIXTURE); assert d['view']=='colorcorrect' and d['time']['last']==48
        assert len(d['nodes'])==8 and len(d['node_data']['roto']['shapes'])==1
        print(json.dumps({'self_check':'ok','fixture_nodes':len(d['nodes']),'frames':48})); return
    if a.generate_inputs:
        w,h=(1920,1080) if a.size=='1080p' else (3840,2160)
        print(json.dumps(make_inputs(a.generate_inputs,w,h,a.frames))); return
    if not a.display: p.error('run in the real window on X11; pass --display for a private Xvfb display')
    if not a.plate_pattern or not a.background_pattern: p.error('supply generated plate and background patterns')
    run_window(a)
if __name__=='__main__': main()
