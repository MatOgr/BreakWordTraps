import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def draw_pose(pose, image):
    drawing = drawing_utils.draw_landmarks(
        image,
        pose.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
    )
    return drawing


def process_frame(image):
    pose_estimation = mp_pose.process(image)
    image_with_pose = draw_pose(pose_estimation, image)
    return image_with_pose


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_pose = process_frame(frame)
        # cv2.imshow("frame", frame_with_pose)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    # cv2.destroyAllWindows()
