from Triangulator import Triangulator
from ExcelWriter import ExcelWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator
from Constants import *

filepath = f"data/videoset{EXAMPLE}/Seq{EXAMPLE}_settings.xlsx"

writer = ExcelWriter(filepath)

cameras = writer.read_params()

# coordinator = Coordinator(writer.write)
coordinator = Coordinator(lambda msg: None)

triangulator = Triangulator(cameras, coordinator.accept)

tracker = BBoxTracker(triangulator.transform)

tracker.start()
