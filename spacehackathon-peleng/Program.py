from Triangulator import Triangulator
from ExcelWriter import ExcelWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator


filepath = "data/videoset1/Seq1_settings.xlsx"

writer = ExcelWriter(filepath)

cameras = writer.read_params()

# coordinator = Coordinator(writer.write)
coordinator = Coordinator(lambda msg: None)

# triangulator = Triangulator(cameras, resolution, coordinator.accept)
triangulator = Triangulator(cameras, coordinator.accept)

tracker = BBoxTracker(triangulator.transform)

tracker.start()
