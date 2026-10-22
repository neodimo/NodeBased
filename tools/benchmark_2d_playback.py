#!/usr/bin/env python3
"""Real-window uncached/cached playback benchmark for the Lane 8 compositing path.

Generated EXR inputs and run JSON belong in the caller-supplied scratch/output directory;
only the small .nbcomp graph and this reproducible generator are tracked.
"""
from __future__ import annotations
import argparse, json, os, pathlib, statistics, subprocess, sys, threading, time, traceback
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
    from PySide6.QtCore import QTimer, Qt
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
    from nodebased.tileexec import TileRegion, SUPPORTED_TILED_KINDS, _all_ancestors
    from nodebased import groups

    events=[]; shown=[]; statuses=[]; worker_profiles=[]; sample_counts={'qt':{},'worker':{}}; node_samples={}; node_costs={}; sample_period_ms=2.0
    lock=threading.Lock(); trace_origin=time.perf_counter(); sample_start=[None]; run_end=[None]; sample_stop=[None]
    def mark(stage, **kw):
        row={'stage':stage,'t_ms':round((time.perf_counter()-trace_origin)*1000,3),**kw}
        with lock: events.append(row)
        if os.environ.get('NB_PLAYBACK_TRACE') and stage in {'playback_start','frame_tick','frame_request','worker_start','compose_start','compose_end','display_conversion','ui_handoff','playback_stop','worker_end'}: print(json.dumps(row),file=sys.stderr,flush=True)
    def sample_stacks():
        while run_end[0] is None:
            sample_time=time.perf_counter()
            if sample_start[0] is None or sample_time < sample_start[0] or sample_time >= sample_stop[0]:
                time.sleep(.002)
                continue
            with lock: frames=sys._current_frames()
            for thread in threading.enumerate():
                if thread.ident not in frames: continue
                lane='qt' if thread is threading.main_thread() else ('worker' if thread.name.startswith('nodebased-preview') else None)
                if lane is None: continue
                current=frames[thread.ident]
                while current is not None:
                    code=current.f_code
                    key=f'{pathlib.Path(code.co_filename).name}:{code.co_name}'
                    sample_counts[lane][key]=sample_counts[lane].get(key,0)+1
                    if lane=='worker' and code.co_name in ('evaluate_raster','_render_tile'):
                        loc=current.f_locals
                        node_id=loc.get('key') if code.co_name=='evaluate_raster' else loc.get('node_id')
                        frame_no=loc.get('frame',0); kind=loc.get('kind')
                        if node_id in doc.get('nodes',{}):
                            kind=kind or doc['nodes'][node_id]['type']; slot=(int(frame_no or 0),node_id,kind,code.co_name)
                            node_samples[slot]=node_samples.get(slot,0)+1
                    current=current.f_back
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
    # This benchmark isolates frame evaluation through the Viewer handoff. The independent
    # timeline label painter can segfault under Mesa/Xvfb while painting frame ticks; it does not
    # participate in that path, so keep that strip out of this headless measurement window.
    win.frame_slider.hide()
    # Xvfb's backing-store paint path is also unstable with sustained full-resolution updates.
    # Evaluation, display conversion and the UI-thread handoff remain active; only rasterizing
    # the already-handed-off widgets is disabled, which is outside this lane's measured boundary.
    win.setUpdatesEnabled(False)
    win.proxy.setCurrentIndex(win.proxy.findData(1))
    win.playback_proxy.setChecked(False)  # measure full-resolution evaluation; automatic proxying changes the gate
    win.show_thumbnails=False
    # Keep the cached run's scene-linear frames warm before the measured playback interval.
    if args.cache_mode=='cached':
        # The normal display-cache budget can retain the complete 48-frame HD sequence but only
        # a 16-frame UHD window. Loop that resident UHD window instead of benchmarking a cache
        # miss immediately after frame 48 wraps to frame 1.
        if args.size=='4k':
            win.dispatcher.document['time'].update(first=33,current=33,last=48)
        warm_doc=win.dispatcher.document; target=warm_doc['view']
        tiled=win.tile_executor.supports_tiled(warm_doc,target)
        warm_frames=range(33,49) if args.size=='4k' else range(1,49)
        for frame in warm_frames:
            key=DisplayCache.key(warm_doc,target,frame,1)
            if tiled:
                region=win.tile_executor.canvas_region(warm_doc,target,frame=frame,tier=1)
                result=win.tile_executor.compose_region(warm_doc,target,region,frame=frame,tier=1)
                pixels=result.pixels; region_key=(region.x,region.y,region.width,region.height)
            else:
                pixels=win.evaluator.evaluate(warm_doc,target,frame=frame,tier=1); region_key=None
            win.display_cache.put(key,pixels,region_key,lazy=True)
        # A UHD float frame is large enough that the ordinary bounded display cache may evict
        # frame 1 while the full 48-frame sequence is being warmed. Verify a recent resident;
        # begin UHD playback there so the measured cached run starts with an actual hit.
        check_frame=33 if args.size=='4k' else 1
        check_key=DisplayCache.key(warm_doc,target,check_frame,1)
        cache_verified=win.display_cache.get(check_key,None,copy=False) is not None
        mark('cache_prewarm_complete',frames=len(warm_frames),verified=cache_verified,display_cache_bytes=win.display_cache.bytes)
        if not cache_verified: raise RuntimeError('display cache did not retain the last prewarmed frame')
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
    original_evaluate=win.evaluator.evaluate
    def evaluate(document,target=None,*a,**kw):
        frame=int(kw.get('frame',document.get('time',{}).get('current',1)))
        before={k:len(v) for k,v in win.evaluator.profile_log.items()}
        t=time.perf_counter(); out=original_evaluate(document,target,*a,**kw)
        flat,flat_target=groups.flatten_groups(document,target or document.get('view'))
        chain=_all_ancestors(flat,flat_target); blocker=next((key for key in chain if flat['nodes'][key]['type'] not in SUPPORTED_TILED_KINDS),None)
        reason=(f"{flat['nodes'][blocker]['type']} has no tile kernel; its coordinate-dependent/full-frame path sets the upstream evaluation path" if blocker else 'full-frame evaluator selected by the caller')
        mark('full_frame_evaluation',frame=frame,target=target,elapsed_ms=round((time.perf_counter()-t)*1000,3),reason=reason)
        for node,rows in win.evaluator.profile_log.items():
            for row in rows[before.get(node,0):]:
                mark('node_evaluation',frame=row['frame'],node=node,kind=flat['nodes'].get(node,{}).get('type','unknown'),
                     path='full-frame fallback',reason=reason,elapsed_ms=round(row['wall_time_ms'],3),cache_hits=row['cache_hits'])
        return out
    win.evaluator.evaluate=evaluate
    original_put=win.display_cache.put
    def put(*a,**kw):
        t=time.perf_counter(); result=original_put(*a,**kw); mark('cache_put',elapsed_ms=round((time.perf_counter()-t)*1000,3),bytes=win.display_cache.bytes); return result
    win.display_cache.put=put
    original_qimage=app_module.to_qimage
    def qimage(*a,**kw):
        t=time.perf_counter(); image=original_qimage(*a,**kw); mark('display_conversion',elapsed_ms=round((time.perf_counter()-t)*1000,3)); return image
    app_module.to_qimage=qimage
    original_media_read=media.read_media_raster
    def read_media(*a,**kw):
        t=time.perf_counter(); out=original_media_read(*a,**kw); mark('plate_read_decode',elapsed_ms=round((time.perf_counter()-t)*1000,3),path=pathlib.Path(a[0]).name if a else ''); return out
    media.read_media_raster=read_media
    def finished(payload,frame,image,status,render_region=None):
        req,cancel=payload
        mark('ui_handoff',frame=req.frame,status=status.split('  ·  ')[0],has_frame=frame is not None)
        if frame is not None:
            shown.append((time.perf_counter(),req.frame)); statuses.append(status.split('  ·  ')[0])
    win.signals.finished.connect(finished)
    original_submit=win.executor.submit
    job_counter=[0]
    def submit(fn,*a,**kw):
        job_counter[0]+=1; job_id=job_counter[0]; job_name=getattr(fn,'__name__','worker')
        def wrapped(*wa,**wk):
            mark('worker_start',job_id=job_id,job=job_name,thread=threading.current_thread().name)
            try: return fn(*wa,**wk)
            finally: mark('worker_end',job_id=job_id,job=job_name,thread=threading.current_thread().name)
        return original_submit(wrapped,*a,**kw)
    win.executor.submit=submit
    sampler=threading.Thread(target=sample_stacks,name='playback-sampler',daemon=True); sampler.start()
    def begin():
        sample_start[0]=time.perf_counter(); sample_stop[0]=sample_start[0]+args.seconds
        mark('playback_start',mode=args.cache_mode,adapter=args.adapter)
        win.toggle_playback(True)
        sample_timer = QTimer(win)
        sample_timer.setSingleShot(True)
        sample_timer.setTimerType(Qt.TimerType.PreciseTimer)
        sample_timer.timeout.connect(finish)
        sample_timer.start(round(args.seconds*1000))
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
    sampler.join()
    # Stop time is the acceptance-window boundary. The worker may finish in-flight frames while
    # draining below, but those tail pictures are outside the measured playback interval.
    measured_shown=[row for row in shown if sample_start[0] <= row[0] <= sample_stop[0]]
    intervals=[(measured_shown[i][0]-measured_shown[i-1][0])*1000 for i in range(1,len(measured_shown))]
    frame_times=[]
    last=None
    sample_start_ms=((sample_start[0]-trace_origin)*1000) if sample_start[0] is not None else None
    sample_stop_ms=((sample_stop[0]-trace_origin)*1000) if sample_stop[0] is not None else None
    for event in events:
        if (sample_start_ms is not None and event['t_ms'] < sample_start_ms
                or sample_stop_ms is not None and event['t_ms'] > sample_stop_ms):
            continue
        if event['stage']=='worker_start': last=event['t_ms']
        elif event['stage']=='ui_handoff' and last is not None and event['has_frame']:
            frame_times.append(event['t_ms']-last); last=None
    flat_doc,flat_target=groups.flatten_groups(doc,doc.get('view')); ancestors=_all_ancestors(flat_doc,flat_target)
    blocker=next((key for key in ancestors if flat_doc['nodes'][key]['type'] not in SUPPORTED_TILED_KINDS),None)
    fallback_reason=(f"{flat_doc['nodes'][blocker]['type']} has no tile kernel; coordinate-dependent evaluation selects the full-frame path" if blocker else None)
    profiled_nodes=[]
    for (frame_no,node_id,kind,code_name),samples in sorted(node_samples.items()):
        path='tile' if code_name=='_render_tile' else 'full-frame fallback'
        profiled_nodes.append({'frame':frame_no,'node':node_id,'kind':kind,'path':path,
                              'fallback_reason':fallback_reason if path=='full-frame fallback' else None,
                              'samples':samples,'estimated_ms':round(samples*sample_period_ms,3)})
        mark('node_profile',frame=frame_no,node=node_id,kind=kind,path=path,samples=samples,
             estimated_ms=round(samples*sample_period_ms,3),fallback_reason=fallback_reason if path=='full-frame fallback' else None)
    result={'case':args.case,'resolution':f'{args.width}x{args.height}','duration_s':args.seconds,
        'cache_mode':args.cache_mode,'adapter':args.adapter,'force_adapter':args.force_adapter,
        'frames_displayed':len(measured_shown),'achieved_fps':round(len(measured_shown)/args.seconds,3),
        'frame_interval_ms':{'p50':percentile(intervals,50),'p95':percentile(intervals,95),'p99':percentile(intervals,99)},
        'worker_to_handoff_ms':{'p50':percentile(frame_times,50),'p95':percentile(frame_times,95),'p99':percentile(frame_times,99)},
        'status_line':statuses[-1] if statuses else win.statusBar().currentMessage(),
        'per_node':sorted(profiled_nodes,key=lambda row:row['estimated_ms'],reverse=True),'timeline':events,
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
