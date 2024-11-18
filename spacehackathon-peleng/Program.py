from Triangulator import Triangulator
from ExcelWriter import ExcelWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator


resolution = (1920, 1072)

filepath = "data/videoset1/Seq1_settings.xlsx"

writer = ExcelWriter(filepath)

cameras = writer.read_params()

# coordinator = Coordinator(lambda msg: writer.write(msg))
coordinator = Coordinator(lambda msg: None)

# triangulator = Triangulator(cameras, resolution, lambda msg: coordinator.accept(msg))
triangulator = Triangulator(cameras, lambda msg: coordinator.accept(msg))

tracker = BBoxTracker(lambda msg: triangulator.transform(msg))

tracker.start()
