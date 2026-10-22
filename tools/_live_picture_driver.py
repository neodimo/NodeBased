import cProfile, io, json, os, pstats, subprocess, sys, threading, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from live_picture_bench import qt_global_to_x11
import tests.isolation
from PySide6.QtCore import QObject, QTimer, Signal, Qt, QPointF, QEvent
from PySide6.QtWidgets import QApplication
import nodebased.gpu3d as gpu3d
_orig_gpu_state=gpu3d._state
force=sys.argv[5]
if force != 'default': gpu3d._state=lambda choice=None:_orig_gpu_state(force if (choice or 'default')=='default' else choice)
from nodebased.app import Window, STYLE
import nodebased.app as app_module
from nodebased.core import load_document
from nodebased.imaging import to_qimage as _to_qimage
from nodebased.tileexec import SUPPORTED_TILED_KINDS, _all_ancestors
from nodebased import groups
from nodebased import media as media_module, paint as paint_module

comp, output, duration, adapter = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
timeline=[]; profiles={}; lock=threading.Lock(); first_drag_move=[None]
def mark(stage, **fields):
    with lock: timeline.append({"stage":stage,"t":time.perf_counter(),**fields})
profile_mode=sys.argv[6]
warm_playback_range = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
main_profile=cProfile.Profile()
if profile_mode=='qt': main_profile.enable()
app=QApplication(sys.argv[:1]); app.setApplicationName("NodeBased"); app.setStyle("Fusion"); app.setStyleSheet(STYLE)
doc=load_document(comp)
plate=os.environ.get('NB_QA_PLATE')
plate=os.path.join(plate, 'plate.%04d.exr') if plate else None
if plate and os.path.exists(plate % 1): doc['nodes']['plate']['params']['path']=plate
else:
    doc['nodes']['plate'].update(type='Constant', params={'width':1920,'height':1080,'red':.12,'green':.3,'blue':.6,'alpha':1.0})
win=Window(doc)
original_handed=win._direct_handed_over
def handed_over(*a,**k): mark('card_upload_and_handoff',accepted=bool(a[0]) if a else None); return original_handed(*a,**k)
win._direct_handed_over=handed_over
class Trace(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.MouseButtonPress,QEvent.Type.MouseMove,QEvent.Type.MouseButtonRelease):
            point = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else None
            mark('mouse_event_received', type=int(event.type()),
                 global_pos=None if point is None else [point.x(), point.y()])
            if (event.type() == QEvent.Type.MouseMove
                    and event.buttons() & Qt.MouseButton.LeftButton
                    and first_drag_move[0] is None):
                first_drag_move[0] = time.monotonic()
                mark('first_drag_move_received')
        if event.type()==QEvent.Type.Paint: mark('paint')
        return False
trace=Trace(); win.viewer.viewport().installEventFilter(trace)
orig_request=win.request_preview
def request(*a,**k):
    mark('preview_requested',tier=k.get('tier_override'),progressive=k.get('progressive',False))
    return orig_request(*a,**k)
win.request_preview=request
original_compose=win.tile_executor.compose_region
def compose(*a,**k):
    before={node:len(rows) for node,rows in win.evaluator.profile_log.items()}
    mark('tier_resample_and_graph_start',tier=k.get('tier')); result=original_compose(*a,**k)
    if not result.tiled:
        document,target,region=a[0],a[1],a[2]
        flat,target=groups.flatten_groups(document,target)
        ancestors=_all_ancestors(flat,target)
        fraction=region.width*region.height/max(1,region.full_width*region.full_height)
        if fraction>=.80 and any(flat['nodes'][key]['type']=='Roto' for key in ancestors):
            mark('tile_fallback_reason',node='Roto',reason='requested region covers at least 80% of the canvas; broad Roto requests use the full-frame evaluator')
    for node,rows in win.evaluator.profile_log.items():
        for row in rows[before.get(node,0):]:
            mark('graph_node_timing',node=node,frame=row['frame'],elapsed_ms=round(row['wall_time_ms'],3),cache_hits=row['cache_hits'])
    mark('graph_evaluation_end',tier=k.get('tier'),tiled=result.tiled,fallback_nodes=list(result.fallback_nodes)); return result
win.tile_executor.compose_region=compose
original_tile=win.tile_executor._render_tile
def render_tile(document,node_id,frame,tier,region,node_digests,cancel):
    node=document['nodes'][node_id]; start=time.perf_counter()
    result=original_tile(document,node_id,frame,tier,region,node_digests,cancel)
    fallback=node['type'] not in SUPPORTED_TILED_KINDS
    mark('graph_node_evaluated',node=node_id,kind=node['type'],tier=tier,
         elapsed_ms=round((time.perf_counter()-start)*1000,3),fallback=fallback,
         reason='kind has no tile kernel' if fallback else None)
    return result
win.tile_executor._render_tile=render_tile
original_evaluate=win.evaluator.evaluate
def evaluate(document,target=None,*a,**k):
    node_id=target or document.get('view'); node=document.get('nodes',{}).get(node_id,{})
    start=time.perf_counter(); result=original_evaluate(document,target,*a,**k)
    mark('graph_node_evaluated',node=node_id,kind=node.get('type','?'),tier=k.get('tier',1),
         elapsed_ms=round((time.perf_counter()-start)*1000,3),fallback=False,reason=None)
    return result
win.evaluator.evaluate=evaluate
original_read_media_raster = media_module.read_media_raster
def read_media_raster(*a, **k):
    started = time.perf_counter()
    mark('source_decode_start', thread=threading.current_thread().name)
    try:
        return original_read_media_raster(*a, **k)
    finally:
        mark('source_decode_end', elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
             thread=threading.current_thread().name)
media_module.read_media_raster = read_media_raster
original_paint_rasterise = paint_module.rasterise
def paint_rasterise(*a, **k):
    started = time.perf_counter()
    frame = a[2] if len(a) > 2 else k.get('frame')
    mark('paint_rasterise_start', frame=frame, thread=threading.current_thread().name)
    try:
        return original_paint_rasterise(*a, **k)
    finally:
        mark('paint_rasterise_end', elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
             frame=frame, thread=threading.current_thread().name)
paint_module.rasterise = paint_rasterise
original_supports=win.tile_executor.supports_tiled
def supports(document,target):
    result=original_supports(document,target)
    if not result:
        flat,target=groups.flatten_groups(document,target); nodes=flat['nodes']
        for node_id in _all_ancestors(flat,target):
            node=nodes[node_id]; reason=None
            if node['type'] not in SUPPORTED_TILED_KINDS: reason='node kind has no tile kernel'
            elif node['type']=='Denoise' and node['params'].get('temporal'): reason='temporal denoise requires full frame'
            elif node['type']=='RotoPaint' and any(float(i.get('patch_blend',0))>0 for i in document.get('node_data',{}).get(node_id,{}).get('items',[]) if i.get('kind')=='stroke'): reason='paint patch blend reads the full painted region'
            elif node.get('input_outputs'): reason='named output layers require full-frame raster'
            elif node['type']=='DirBlur' and node['params'].get('blur_type','linear')!='linear': reason='zoom/radial blur reads the full frame'
            if reason: mark('tile_fallback_reason',node=node_id,kind=node['type'],reason=reason); break
    return result
win.tile_executor.supports_tiled=supports
original_execute=win.dispatcher.execute
def execute(*a,**k): mark('document_edit_applied'); return original_execute(*a,**k)
win.dispatcher.execute=execute
original_submit=win.executor.submit
def submit(fn,*a,**k):
    def run():
        mark('worker_start'); prof=cProfile.Profile() if profile_mode=='worker' else None
        if prof: prof.enable()
        try: return fn(*a,**k)
        finally:
            if prof:
                prof.disable()
                with lock: profiles.setdefault(threading.get_ident(),[]).append(prof)
    return original_submit(run)
win.executor.submit=submit
original_cache_get=win.display_cache.get
def cache_get(*a,**k):
    mark('display_cache_lookup_start'); value=original_cache_get(*a,**k)
    mark('display_cache_lookup_end',hit=value is not None); return value
win.display_cache.get=cache_get
original_decode_get=win.decode_pool.get
def decode_get(key):
    value=original_decode_get(key)
    mark('decode_ahead_lookup',hit=value is not None)
    return value
win.decode_pool.get=decode_get
def traced_qimage(*a,**k):
    mark('to_qimage_start'); result=_to_qimage(*a,**k); mark('to_qimage_end'); return result
app_module.to_qimage=traced_qimage
class Bridge(QObject): call=Signal(object)
bridge=Bridge(); bridge.call.connect(lambda f:f(), Qt.ConnectionType.QueuedConnection)
def on_main(fn):
    box={}; ev=threading.Event()
    def run():
        try: box['v']=fn()
        finally: ev.set()
    bridge.call.emit(run); ev.wait(10); return box.get('v')
def xd(*args): subprocess.run(['xdotool',*map(str,args)],env={**os.environ,'DISPLAY':os.environ['DISPLAY']},check=True)
def busy(): return bool(win.busy) or len(win.preview_queue)>0
def settle(timeout=90, stable_for=.35):
    until=time.monotonic()+timeout; stable=None
    while time.monotonic()<until:
        ready=(not busy() and win.frame is not None and on_main(lambda:win.viewer.last_scale)==int(on_main(lambda:win.proxy.currentData())))
        now=time.monotonic()
        if ready:
            if stable is None: stable=now
            elif now-stable>=stable_for: return
        else: stable=None
        time.sleep(.01)
def wait_new(generation, timeout=30):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if win.frame_generation>generation: return time.monotonic()
        time.sleep(.002)
    return None
def scenario():
  try:
    mark('scenario_start'); settle(); on_main(lambda:(win.graph.items_by_id['roto'].setSelected(True),win.set_viewer_roi(on=False))); settle()
    mark('series_start',series='roto_drag')
    shape=win.dispatcher.document['node_data']['roto']['shapes'][0]['points'][0]
    def xy():
      p=win.viewer.viewport().mapToGlobal(win.viewer.mapFromScene(QPointF(shape['x'],shape['y'])))
      return p.x(),p.y()
    logical_x,logical_y=on_main(xy)
    ratio=float(win.devicePixelRatioF())
    x,y=qt_global_to_x11((logical_x,logical_y),ratio)
    mark('drag_coordinates', logical=[logical_x,logical_y], x11=[x,y],
         device_pixel_ratio=ratio, viewport=[win.viewer.viewport().width(),win.viewer.viewport().height()])
    xd('mousemove',x,y); time.sleep(.25)
    g0=win.frame_generation; start=time.monotonic(); first=None
    xd('mousedown',1); time.sleep(.08)
    for n in range(1,7):
      # Keep each move distinct and give Qt time to consume it while the button is
      # still held. A burst of xdotool moves can be coalesced into one event at
      # release, which tests the input queue rather than live preview delivery.
      xd('mousemove',x+round(12*ratio)*n,y+round(8*ratio)*n)
      if first is None: first=wait_new(g0,.045)
      time.sleep(.16)
    drag_end=time.monotonic(); frames=win.frame_generation-g0; xd('mouseup',1)
    if first is None: first=wait_new(g0,45)
    settle()
    first_move = first_drag_move[0]
    drag={"series":"roto_drag","adapter":adapter,
          "first_picture_s":None if first is None else round(first-start,4),
          "first_move_to_picture_s":None if first is None or first_move is None else round(first-first_move,4),
          "first_move_received_s":None if first_move is None else round(first_move-start,4),
          "picture_during_drag":first is not None and first<drag_end,"frames_during_drag":frames,
          "settle_s":round(time.monotonic()-drag_end,4),"status":win.viewer_info.text()}
    mark('series_end',series='roto_drag'); mark('series_start',series='uncached_scrub')
    g=win.frame_generation; xd('key','Right'); scrub_start=time.monotonic(); scrub_first=wait_new(g,60); settle()
    scrub={"series":"uncached_scrub","adapter":adapter,"first_picture_s":None if scrub_first is None else round(scrub_first-scrub_start,4),"settle_s":round(time.monotonic()-scrub_start,4),"status":win.viewer_info.text()}
    mark('series_end',series='uncached_scrub')
    on_main(lambda:(win.proxy.setCurrentIndex(win.proxy.findData(4)),win.playback_proxy.setChecked(False))); settle()
    warmed_frames = 0; cache_warm_s = None
    if warm_playback_range:
      mark('series_start',series='tier4_cache_warm')
      warm_start = time.monotonic()
      time_range = dict(win.dispatcher.document['time'])
      for frame_number in range(int(time_range['first']), int(time_range['last']) + 1):
        generation = win.frame_generation
        on_main(lambda f=frame_number: win.set_time(transient=True, current=f))
        if wait_new(generation, 90) is None: break
        settle(timeout=90, stable_for=.02)
      on_main(lambda f=int(time_range['first']): win.set_time(transient=True, current=f))
      settle(timeout=90, stable_for=.02)
      current_doc = win.dispatcher.document
      target = current_doc.get('view')
      warm_frames = range(int(time_range['first']), int(time_range['last']) + 1)
      identity = win.display_cache.identity(current_doc, target, 4)
      # DisplayCache entries are region-keyed. Playback can use the same selected-tier
      # frame from a viewport region, so get(key, region=None) would incorrectly report
      # every such entry missing. resident_frames checks the exact content identity and
      # frame while intentionally ignoring the presentation region.
      warmed_frames = len(win.display_cache.resident_frames(identity, warm_frames))
      cache_warm_s = round(time.monotonic() - warm_start, 3)
      mark('playback_range_warmed', requested_frames=int(time_range['last'])-int(time_range['first'])+1,
           cached_frames=warmed_frames, elapsed_s=cache_warm_s)
      mark('series_end',series='tier4_cache_warm')
    mark('series_start',series='tier4_playback')
    playback_timeline_start = time.perf_counter()
    t=time.monotonic(); on_main(lambda:win.toggle_playback(True)); observations=[]
    while time.monotonic()-t<duration:
        observations.append((time.monotonic()-t,win.dispatcher.document['time']['current'],win.frame_generation))
        time.sleep(.01)
    on_main(lambda:win.toggle_playback(False)); elapsed=time.monotonic()-t
    playback_timeline_end = time.perf_counter()
    mark('series_end',series='tier4_playback')
    playback_events = [event for event in timeline
                       if playback_timeline_start <= event.get('t', 0) <= playback_timeline_end]
    playback_cache_lookups = [event for event in playback_events
                               if event.get('stage') == 'display_cache_lookup_end']
    playback_graph_evaluations = sum(event.get('stage') == 'graph_evaluation_end'
                                     for event in playback_events)
    frame_count=0; last_generation=observations[0][2] if observations else -1
    last_frame=observations[0][1] if observations else None
    for _stamp,frame,generation in observations[1:]:
        if generation>last_generation:
            if frame!=last_frame: frame_count+=1; last_frame=frame
            last_generation=generation
    playback={"series":"tier4_playback","adapter":adapter,"fps":round(frame_count/max(elapsed,.001),3),"rendered_frame_changes":frame_count,"elapsed_s":round(elapsed,3),"fps_target":win.dispatcher.document['time']['fps'],"playback_range_warmed":warm_playback_range,"cached_range_frames":warmed_frames,"cache_warm_s":cache_warm_s,"display_cache_lookups":len(playback_cache_lookups),"display_cache_hits":sum(event.get('hit') is True for event in playback_cache_lookups),"graph_evaluations":playback_graph_evaluations,"status":win.viewer_info.text()}
    settle(60)
    if profile_mode=='qt': main_profile.disable()
    profile_rows=[]
    selected=([(threading.get_ident(),[main_profile])] if profile_mode=='qt' else list(profiles.items()))
    for tid,profs in selected:
      stats=pstats.Stats(profs[0])
      for prof in profs[1:]: stats.add(prof)
      txt=io.StringIO(); stats.stream=txt; stats.sort_stats('cumtime').print_stats(50)
      profile_rows.append({'thread_id':tid,'top':txt.getvalue()})
    payload={'adapter':adapter,'results':[drag,scrub,playback],'timeline':timeline,'profiles':profile_rows}
    with open(output,'w') as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,ensure_ascii=False),flush=True)
  except BaseException:
    import traceback; traceback.print_exc()
  finally: os._exit(0)
screen = app.primaryScreen()
# Window geometry is in Qt logical pixels. Filling a fixed 1920x1080 logical rect
# made the window 3840x2160 physical pixels at QT_SCALE_FACTOR=2, leaving the
# benchmark's target handle outside the visible Xvfb screen.
win.setGeometry(screen.geometry() if screen is not None else win.geometry())
win.show(); win.raise_(); win.activateWindow(); QTimer.singleShot(1000,lambda:threading.Thread(target=scenario,daemon=True).start()); app.exec()
