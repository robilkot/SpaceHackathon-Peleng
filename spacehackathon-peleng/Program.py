from Triangulator import Triangulator
from CoordsWriter import CoordsWriter
from BBoxTracker import BBoxTracker
from Coordinator import Coordinator

# Set up constants

cameras = {
    # todo
}

resolution = (1920, 1072)  # todo move to camera class

filepath = "data/videoset0/Seq0_settings.xlsx"

# Init services

writer = CoordsWriter(filepath)

coordinator = Coordinator(lambda msg: writer.write(msg))

triangulator = Triangulator(cameras, resolution, lambda msg: coordinator.accept(msg))

tracker = BBoxTracker(lambda msg: triangulator.transform(msg))

tracker.start()
