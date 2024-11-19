from Triangulator import Triangulator
from ExcelWriter import ExcelWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator
from Constants import *


writer = ExcelWriter(SETTINGS)

cameras = writer.read_params()

coordinator = Coordinator(writer.write)
# coordinator = Coordinator(lambda msg: None)

triangulator = Triangulator(cameras, coordinator.accept)

tracker = BBoxTracker(lambda msg: None)

tracker.start()
