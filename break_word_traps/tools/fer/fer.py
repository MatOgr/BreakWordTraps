import time
import warnings

import cv2
import numpy as np
import torch
from mediapipe.python.solutions.face_mesh import FaceMesh
from PIL import Image

from .models import LSTMPyTorch, ResNet50
from .sub_functions import get_box, pth_processing
from .types import Emotion

warnings.simplefilter("ignore", UserWarning)


### Models
name_backbone_model = "FER_static_ResNet50_AffectNet.pt"
# name_LSTM_model = 'IEMOCAP'
# name_LSTM_model = 'CREMA-D'
# name_LSTM_model = 'RAMAS'
# name_LSTM_model = 'RAVDESS'
# name_LSTM_model = 'SAVEE'
name_LSTM_model = "Aff-Wild2"

# torch
DICT_EMO = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
}


class FER:
    def __init__(self, name_backbone_model, name_LSTM_model) -> None:
        self.load_models(name_backbone_model, name_LSTM_model)

    def load_models(self, name_backbone_model, name_LSTM_model):
        pth_backbone_model = ResNet50(7, channels=3)
        pth_backbone_model.load_state_dict(torch.load(name_backbone_model))
        self.backbone = pth_backbone_model

        pth_LSTM_model = LSTMPyTorch()
        pth_LSTM_model.load_state_dict(
            torch.load("FER_dinamic_LSTM_{0}.pt".format(name_LSTM_model))
        )
        self.lstm = pth_LSTM_model

    def eval(self):
        self.backbone.eval()
        self.lstm.eval()

    def analyse_image(self, frame) -> Emotion | None:
        with FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            frame_copy = frame.copy()
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            emotion = self.retrieve_emotions(frame_copy, face_mesh)
        return emotion

    def retrieve_emotions(
        self, image: np.ndarray, face_mesh: FaceMesh
    ) -> Emotion | None:
        lstm_features = []
        start = time.time()
        results = face_mesh.process(image)
        image.flags.writeable = True
        # logger.debug(results.multi_face_landmarks)
        # logger.debug(results)
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                width, height = image.shape
                startX, startY, endX, endY = get_box(fl, width, height)
                cur_face = image[startY:endY, startX:endX]

                cur_face = pth_processing(Image.fromarray(cur_face))
                features = (
                    torch.nn.functional.relu(
                        self.pth_backbone_model.extract_features(cur_face)
                    )
                    .detach()
                    .numpy()
                )

                if len(lstm_features) == 0:
                    lstm_features = [features] * 10
                else:
                    lstm_features = lstm_features[1:] + [features]

                lstm_f = torch.from_numpy(np.vstack(lstm_features))
                lstm_f = torch.unsqueeze(lstm_f, 0)
                output = self.pth_LSTM_model(lstm_f).detach().numpy()

                cl = np.argmax(output)
                # logger.debug(f"{cl =}")
                label = DICT_EMO[cl]
                # logger.debug(f"{label = }")
                # text = label + " {0:.1%}".format(output[0][cl])
                result = label

        else:
            result = None
        processing_time = time.time() - start
        # logger.debug(f"Inference time: {processing_time}")
        return result
