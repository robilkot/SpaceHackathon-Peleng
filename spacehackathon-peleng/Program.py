from Triangulator import Triangulator
from ExcelWriter import ExcelWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator


resolution = (1920, 1072)

filepath = "data/videoset0/Seq0_settings.xlsx"

writer = ExcelWriter(filepath)

cameras = writer.read_params()

coordinator = Coordinator(lambda msg: writer.write(msg))

triangulator = Triangulator(cameras, resolution, lambda msg: coordinator.accept(msg))

tracker = BBoxTracker(lambda msg: triangulator.transform(msg))

tracker.start()
