import json, os, subprocess, sys, threading, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tests.isolation
from PySide6.QtCore import QObject, QTimer, Signal, Qt, QPointF
from PySide6.QtWidgets import QApplication
from nodebased.app import Window, STYLE
from nodebased.core import load_document

comp, output, duration, adapter = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
app=QApplication(sys.argv[:1]); app.setApplicationName("NodeBased"); app.setStyle("Fusion"); app.setStyleSheet(STYLE)
doc=load_document(comp)
plate=QA_PLATE=os.path.join(os.path.dirname(os.path.dirname(comp)), 'plate', 'plate.%04d.exr')
if os.path.exists(plate % 1): doc['nodes']['plate']['params']['path']=plate
else:
    doc['nodes']['plate'].update(type='Constant', params={'width':1920,'height':1080,'red':.12,'green':.3,'blue':.6,'alpha':1.0})
win=Window(doc)
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
def settle(timeout=90):
    until=time.monotonic()+timeout
    while time.monotonic()<until and (busy() or win.frame is None): time.sleep(.02)
    time.sleep(.25)
def wait_new(generation, timeout=30):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if win.frame_generation>generation: return time.monotonic()
        time.sleep(.002)
    return None
def scenario():
  try:
    settle(); on_main(lambda:(win.graph.items_by_id['roto'].setSelected(True),win.set_viewer_roi(on=False))); settle()
    shape=win.dispatcher.document['node_data']['roto']['shapes'][0]['points'][0]
    def xy():
      p=win.viewer.viewport().mapToGlobal(win.viewer.mapFromScene(QPointF(shape['x'],shape['y'])))
      return p.x(),p.y()
    x,y=on_main(xy); xd('mousemove',x,y); time.sleep(.25)
    g0=win.frame_generation; start=time.monotonic(); first=None
    xd('mousedown',1); time.sleep(.04)
    for n in range(1,7):
      xd('mousemove',x+12*n,y+8*n)
      if first is None: first=wait_new(g0,.045)
      time.sleep(.035)
    drag_end=time.monotonic(); frames=win.frame_generation-g0; xd('mouseup',1)
    if first is None: first=wait_new(g0,45)
    settle()
    drag={"series":"roto_drag","adapter":adapter,"first_picture_s":None if first is None else round(first-start,4),"picture_during_drag":first is not None and first<drag_end,"frames_during_drag":frames,"settle_s":round(time.monotonic()-drag_end,4),"status":win.viewer_info.text()}
    g=win.frame_generation; xd('key','Right'); scrub_start=time.monotonic(); scrub_first=wait_new(g,60); settle()
    scrub={"series":"uncached_scrub","adapter":adapter,"first_picture_s":None if scrub_first is None else round(scrub_first-scrub_start,4),"settle_s":round(time.monotonic()-scrub_start,4),"status":win.viewer_info.text()}
    on_main(lambda:(win.proxy.setCurrentIndex(win.proxy.findData(4)),win.playback_proxy.setChecked(False))); settle()
    t=time.monotonic(); g=win.frame_generation; on_main(lambda:win.toggle_playback(True)); time.sleep(duration); on_main(lambda:win.toggle_playback(False)); elapsed=time.monotonic()-t
    playback={"series":"tier4_playback","adapter":adapter,"fps":round((win.frame_generation-g)/max(elapsed,.001),3),"elapsed_s":round(elapsed,3),"status":win.viewer_info.text()}
    with open(output,'w') as f: json.dump({"adapter":adapter,"results":[drag,scrub,playback]},f,indent=2)
    print(json.dumps({"adapter":adapter,"results":[drag,scrub,playback]},ensure_ascii=False),flush=True)
  except BaseException:
    import traceback; traceback.print_exc()
  finally: app.quit()
win.setGeometry(0,0,1920,1080); win.show(); win.raise_(); win.activateWindow(); QTimer.singleShot(1000,lambda:threading.Thread(target=scenario,daemon=True).start()); app.exec()
