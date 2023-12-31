"""
繪製臉部網格以及眉毛、眼睛
"""
# 匯入 OpenCV-python 庫
import cv2

# 匯入 MediaPipe 庫，用於人臉偵測和網格繪製
import mediapipe as mp
import sys

# 設定繪製網格點和連接線等標記的工具
mp_drawing = mp.solutions.drawing_utils
# mediapipe 繪圖樣式
mp_drawing_styles = mp.solutions.drawing_styles
# 引入人臉網格偵測功能，其中屬性 face_mesh 是一個用來識別並繪製人臉高精度網格點的模型
mp_face_mesh = mp.solutions.face_mesh

"""以上相同"""

# 自訂繪圖參數：調用 draw_face_annotations() 函數時可使用預設值，也可傳入自訂的參數
MY_DRAWING_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  # 整數


# 封裝繪製臉部標示的函數
def draw_face_annotations(_image, _landmarks, _drawing_spec=None):
    # 繪製網格
    mp_drawing.draw_landmarks(
        image=_image,
        landmark_list=_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        # 預設為 None
        landmark_drawing_spec=_drawing_spec,
        connection_drawing_spec=mp_drawing_styles.
        get_default_face_mesh_tesselation_style(),
    )

    # 繪製輪廓
    mp_drawing.draw_landmarks(
        image=_image,
        landmark_list=_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=_drawing_spec,
        connection_drawing_spec=mp_drawing_styles.
        get_default_face_mesh_contours_style(),
    )

    # 繪製眼睛
    mp_drawing.draw_landmarks(
        image=_image,
        landmark_list=_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=_drawing_spec,
        connection_drawing_spec=mp_drawing_styles.
        get_default_face_mesh_iris_connections_style(),
    )


# 初始化攝像頭：index=0，設定為第一個設備
cap = cv2.VideoCapture(0)

# 啟用人臉網格偵測，設定相關參數
with mp_face_mesh.FaceMesh(
    # 一次偵測最多幾個人臉
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    if not cap.isOpened():
        print("無法開啟攝像頭")
        sys.exit(1)

    while True:
        success, img = cap.read()
        if not success:
            print("無法開啟攝像頭")
            break

        # 顏色 BGR 轉換為 RGB
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 取得人臉網格資訊
        results = face_mesh.process(img2)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 調用自訂函數
                # 使用預設值
                draw_face_annotations(img, face_landmarks)
                # 使用自訂參數設定值
                # draw_face_annotations(img, face_landmarks, MY_DRAWING_SPEC)

        # 顯示影像並設置標題
        cv2.imshow("Example", img)
        # 檢查是否有按下'ESC'、'q'鍵或關閉視窗
        key = cv2.waitKey(1) & 0xFF
        if (
            key == 27
            or key == ord("q")
            or cv2.getWindowProperty("Example", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

cap.release()
cv2.destroyAllWindows()
