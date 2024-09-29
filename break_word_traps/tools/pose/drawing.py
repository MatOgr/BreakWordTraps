import cv2
import mediapipe as mp


class PoseDrawer:
    def __init__(self):
        self.pose = mp.solutions.pose
        self.pose_connections = self.pose.POSE_CONNECTIONS
        self.drawing_styles = mp.solutions.drawing_styles
        self.drawing_utils = mp.solutions.drawing_utils

    def draw_pose(self, pose_estimation, image):
        drawing = self.drawing_utils.draw_landmarks(
            image,
            pose_estimation.pose_landmarks,
            self.pose_connections,
            landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style(),
        )
        return drawing

    def process_frame(self, pose, image):
        pose_estimation = pose.process(image)
        image_with_pose = self.draw_pose(pose_estimation, image)
        return image_with_pose

    def process_video(self, video_file):
        # TODO: Add support for video processing
        with cv2.VideoCapture(video_file) as cap:
            with self.pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.7,
            ) as pose:
                success, frame = cap.read()
                frames = []
                while success:
                    frame_with_pose = self.process_frame(pose, frame)
                    frames.append(frame_with_pose)
                    # cv2.imshow("frame", frame_with_pose)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    success, frame = cap.read()
        return frames
        # cv2.destroyAllWindows()
