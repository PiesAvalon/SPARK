
import numpy as np




if __name__ == "__main__":
    from network.cv.yolov4.main_new import YOLOV4CspDarkNet53_ms
    import mindspore
    net=YOLOV4CspDarkNet53_ms()
    images = mindspore.Tensor(np.random.randn(1, 3, 416, 416), dtype=mindspore.float32)
    data= net(images)


