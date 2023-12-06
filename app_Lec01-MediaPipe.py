'''
1. 導入所需的函式庫
'''
# 導入所需的套件
# OpenCV
import cv2
# MediaPipe
import mediapipe as mp
# 系統功能
import sys
'''
2. 初始化：攝像頭、模型與工具
'''
# 初始化攝像頭：index=0，設定為第一個設備
cap = cv2.VideoCapture(0)
# 初始化人臉偵測功能：使用 face_detection 模型
mp_face_detection = mp.solutions.face_detection
# 初始化繪圖工具
mp_drawing = mp.solutions.drawing_utils
'''
3. 進行偵測
'''
# 初始化 FaceDetection 類的對象
with mp_face_detection.FaceDetection(
    # 表示選擇特定的人臉偵測模型
    model_selection=0, # =0, 使用預設的模型
    # 偵測的最小信心閾值，越高代表對偵測結果要求更嚴格
    min_detection_confidence=0.5) as face_detection:
    #    pass
        # 以下先空白 ...
#######################

        # 檢查攝像頭是否開啟
        if not cap.isOpened():
            print("無法開啟攝像頭")
            # 結束程序：使用退出代碼「1」表示為異常退出
            sys.exit(1)        

#######################

        # 假如攝像頭是開啟狀態
        # 透過附帶條件的無線迴圈捕捉影像
        while True:
            # 讀取影像，返回兩個值，第一個是布林值代表是否成功
            success, img = cap.read()
            # 假如不成功
            if not success:
                print("無法獲取畫面")
                # 中斷迴圈
                break
            # 假如成功 ....
            # 接續以下程序

#######################

        # 轉換為 MediaPipe 可處理的格式
        # 將 writeable 屬性設置為 False 
        # 要將影像數據傳給函數進行大量處理或轉換時使用
        # 有助於提高性能和降低內存使用
        img.flags.writeable = False

        # 轉換格式 COLOR_BGR2RGB 
        # 要先讀說明書確定每個module需要的是哪種格式的影像；
        # 目前的模型需要的影像為grb => BRG2RGB
        # 但opencv需使用bgr的格式，所以要轉回來給opencv讀取 => RGB2BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 進行人臉偵測
        results = face_detection.process(img)

        # 轉換為 OpenCV 可處理的格式
        img.flags.writeable = True
        # 轉換格式 COLOR_RGB2BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 如果有偵測到人臉，進行標記
        if results.detections:
            # 輸出查看一下結果數
            print(f'偵測到 {len(results.detections)} 張臉')
            # 遍歷
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

        # 顯示影像並設置標題
        cv2.imshow('Example', img)
        
        # 檢查是否有按下'ESC'、'q'鍵或關閉視窗
        key = cv2.waitKey(1) & 0xFF
        if (
            key == 27
            or key == ord("q")
            or cv2.getWindowProperty("Example", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

#############
#############
except Exception as e:
    pass
#############
#############

'''
4. 結束程序：釋放資源並關閉視窗
'''
# 釋放攝像頭資源
cap.release()
# 關閉 OpenCV 視窗
cv2.destroyAllWindows()