import numpy as np
import onnxruntime as ort
from .image import InferenceHandler
# from .openVinoCore import Network
from .model_utility import transform_image_to_framework,\
                           invert_transform
from time import perf_counter


class GeneralModel():
    def __init__(self, model_path, framework, overlaps, input_shape=None) -> None:
        self.model_path = model_path
        self.framework = framework

        assert len(overlaps) <= 2, \
            "Overlaps argument has too many element, expected 2 for x and y"

        self.overlaps = overlaps
        self.input_shape = input_shape
        if framework == "ov":
            self.init_ov_model()
        elif framework == "tf":
            self.init_tf_model()
        elif framework == "onnx":
            self.init_onnx_model()

    def update_inference_handler(self, img):
        self.inference_handler = InferenceHandler(img,
                                                  self.input_shape,
                                                  overlaps=self.overlaps)

    def init_onnx_model(self):
        print(f'ort avail providers: {ort.get_available_providers()}')
        try:
            self.network = ort.InferenceSession(
                str(self.model_path.with_suffix(".onnx")),
                providers=["CUDAExecutionProvider"])
            self.execProvider="CUDA"
        except RuntimeError:
            self.network = ort.InferenceSession(
                str(self.model_path.with_suffix(".onnx")),
                providers=["CPUExecutionProvider"])
            self.execProvider = "defaultCPU"
        self.input_layer_shape = self.network.get_inputs()[0].shape[1:3]
        self.input_shape = self.input_layer_shape

    def init_tf_model(self):
        self.network = adjust_model_input(self.model_path, self.input_shape)

    def init_ov_model(self):
        self.xml_path = str(self.model_path.with_suffix(".xml"))
        self.bin_path = str(self.model_path.with_suffix(".bin"))
        self.network = Network(xml_path=self.xml_path, bin_path=self.bin_path, batch_size=1)
        self.network.load_model()
        self.input_layer_shape = self.network.input_layer_shape
        self.input_shape = self.network.input_layer_shape


def predict(image, model: GeneralModel):
    model.update_inference_handler(image)
    model.inference_handler.get_crop_stack()
    t1 = perf_counter()
    shift = image[image != 0].min()
    max_value = image.max()
    for i, img in enumerate(model.inference_handler.img_stack):
        # start = perf_counter()
        
        adapted_img = transform_image_to_framework(
            framework=model.framework,
            image=img,
            max_value=max_value,
            shift=shift,
            channels=1)

        if model.framework == "ov":
            model.network.synchronous_inference(adapted_img)
            output_name = list(model.network.extract_output().keys())[0]
            output = model.network.extract_output()[output_name]

            # in newer Versions of Ov the buffer attribute needs to be called 
            # to get the output rather than getting it directly
            if not isinstance(output, np.ndarray):
                output = output.buffer
        elif model.framework == "tf":
            output = model.network.predict(adapted_img)
        elif model.framework == "onnx":
            input_name = model.network.get_inputs()[0].name
            output_name = model.network.get_outputs()[0].name
            output = model.network.run(
                [output_name], {input_name: adapted_img})[0]

        output_img = invert_transform(
            framework=model.framework,
            tensor=output,
            img_shape=img.shape,
            max_value=255)

        # cv2.imwrite("D:/debug.png", output_img)
        model.inference_handler.set_stack_element(
            prediction=output_img, index=i)
        # end = perf_counter()
        # patches_stats.append(end-start)
    t2 = perf_counter()
    print("inferencetime: ",
          t2 - t1, "for ",
          model.inference_handler.crop_shape,
          len(model.inference_handler))
    inferenceTime = t2-t1
    return model.inference_handler.join_pred(),\
        model.execProvider,\
        inferenceTime
