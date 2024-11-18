from Triangulator import Triangulator
from CoordsWriter import CoordsWriter
from BBoxTracker import BBoxTracker

# Set up constants

cameras = {
    # todo
}

resolution = (1920, 1072)  # todo move to camera class

filepath = "data/videoset0/Seq0_settings.xlsx"

# Init services

writer = CoordsWriter(filepath)

triangulator = Triangulator(cameras, resolution, lambda msg: writer.write(msg))

tracker = BBoxTracker(lambda msg: triangulator.transform(msg))

tracker.start()
