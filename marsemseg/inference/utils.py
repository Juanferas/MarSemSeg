import hydra
import cv2


def read_config(folder: str = "../", filename: str = "params"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=folder, version_base=None)
    cfg = hydra.compose(config_name=filename)
    return cfg


def get_fps(video: cv2.VideoCapture) -> int:
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

    if int(major_ver) < 3:
        fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
        print(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
        )

    else:
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    return int(fps)
