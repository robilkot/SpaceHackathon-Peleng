RESOLUTION: tuple[int, int] = (1920, 1072)
TIMESTEP: float = 0.5
CAMERAS_COUNT: int = 3
START_FRAME: int = 20
EXAMPLE = 4
CALL_PREDICTOR = 5
LEFTED_FRAMES = 10
PATH_RGB = 'data/videoset{}/Seq{}_camera{}.mov'
PATH_IR = 'data/videoset{}/Seq{}_camera{}T.mov'
SETTINGS = f"data/videoset{EXAMPLE}/Seq{EXAMPLE}_settings.xlsx"

DEBUG_WRITER = False
DEBUG_TRACKER = False
DEBUG_TRIANGULATOR = True
DEBUG_COORDINATOR = True

SHOW_TRACKER = True
