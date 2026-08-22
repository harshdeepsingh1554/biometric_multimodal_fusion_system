 
import os
import logging
import onnxruntime as ort
 
logger = logging.getLogger(__name__)
 
 
class ArcFaceONNXModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace ONNX model weights missing at: {model_path}")
        logger.info(f"Loading ArcFace ONNX model graph: {model_path}")

        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")


        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Model loaded. Providers in use: {self.session.get_providers()}")


    def forward(self, tensor):

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tensor}
        )

        return outputs[0]