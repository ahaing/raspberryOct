"""
繪製臉部網格以及眉毛、眼睛
"""
# 匯入 OpenCV-python 庫
import cv2

# 匯入 MediaPipe 庫，用於人臉偵測和網格繪製
import mediapipe as mp
import sys
import numpy as np

# 定義嘴巴上下唇的標記索引
UPPER_LIP = [61, 40, 37, 0, 267, 269, 270, 409]
LOWER_LIP = [291, 375, 321, 405, 314, 17, 84, 181]

# 左右眼的上下眼瞼特徵的索引
LEFT_EYE_UPPER = [386, 374, 373, 390, 388, 466]
LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380]
RIGHT_EYE_UPPER = [159, 145, 144, 163, 161, 246]
RIGHT_EYE_LOWER = [33, 133, 163, 144, 145, 153]

# 設定繪製網格點和連接線等標記的工具
mp_drawing = mp.solutions.drawing_utils
# mediapipe 繪圖樣式
mp_drawing_styles = mp.solutions.drawing_styles

# 引入人臉網格偵測功能，其中屬性 face_mesh 是一個用來識別並繪製人臉高精度網格點的模型
mp_face_mesh = mp.solutions.face_mesh

# 自訂繪圖參數：調用 draw_face_annotations() 函數時可使用預設值，也可傳入自訂的參數
# 兩個參數都是整數
MY_DRAWING_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# 自訂函數：檢測眼睛是否閉合
def is_eye_closed(eye_upper, eye_lower, landmarks, threshold=0.02):
    upper_points = [landmarks.landmark[i] for i in eye_upper]
    lower_points = [landmarks.landmark[i] for i in eye_lower]

    distance = sum(
        [abs(upper.y - lower.y) for upper, lower in zip(
            upper_points, lower_points
        )]
    ) / len(upper_points)
    return distance < threshold


# 偵測嘴巴是否開啟
def is_mouth_open(face_landmarks):
    # 計算嘴唇開合的平均距離
    mouth_open = 0
    for i in range(len(UPPER_LIP)):
        mouth_open += abs(
            face_landmarks.landmark[UPPER_LIP[i]].y
            - face_landmarks.landmark[LOWER_LIP[i]].y
        )

    mouth_open /= len(UPPER_LIP)
    # print("嘴形閉合的閥值：", mouth_open)
    # 根據實際情況設定閾值，
    return mouth_open > 0.03


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
        #
        # 設置視窗尺寸為 寬x高 800x600
        img = cv2.resize(img, (800, 600))
        # 繪製 800x600 的黑色畫布
        # 特別注意 zeros 的參數是先 column(高) 然後 row(寬)，所以順序相反
        # 3 代表標準的 RGB 顏色模型，即紅色、綠色和藍色
        # dtype 指定陣列中數據的類型，uint8 表示範圍從 0 到 255
        black_cover = np.zeros((600, 800, 3), dtype='uint8')

        # 顏色 BGR 轉換為 RGB
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 取得人臉網格資訊
        results = face_mesh.process(img2)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 調用自訂函數：傳入布幕以及特徵
                draw_face_annotations(black_cover, face_landmarks)
                # 左右眼分開判斷
                left_eye_closed = is_eye_closed(
                    LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                    face_landmarks, threshold=0.005
                )
                right_eye_closed = is_eye_closed(
                    RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                    face_landmarks, threshold=0.005
                )

                # 輸出結果
                # 這裡僅是將結果輸出，在應用的專案中可連動其他邏輯程序
                if left_eye_closed:
                    print("左眼閉上")
                else:
                    print("左眼張開")

                if right_eye_closed:
                    print("右眼閉上")
                else:
                    print("右眼張開")
                # 調用自訂函數
                # 調用自訂函數 is_mouth_open
                if is_mouth_open(face_landmarks):
                    print("張嘴")
                else:
                    print("閉嘴")

                # 使用預設值
                draw_face_annotations(img, face_landmarks)
                # 使用自訂參數設定值
                # draw_face_annotations(img, face_landmarks, MY_DRAWING_SPEC)

        # 顯示影像並設置標題
        cv2.imshow("Black Cover", black_cover)
        # 檢查是否有按下'ESC'、'q'鍵或關閉視窗
        key = cv2.waitKey(1) & 0xFF
        if (
            key == 27
            or key == ord("q")
            or cv2.getWindowProperty("Black Cover", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

cap.release()
cv2.destroyAllWindows()
