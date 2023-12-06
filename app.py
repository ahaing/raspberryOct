# 載入庫
import cv2
from cv_utils import should_exit

# 讀入 Haar cascades 臉部檢測模型
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# 開啟攝像頭
cap = cv2.VideoCapture(0)
# 設定影像寬度
cap.set(3, 640)
# 設定影像高度
cap.set(4, 480)
# 持續讀取攝像頭影像，直到 'ESC' 鍵被按下或程式被終止
while True:
    ret, img = cap.read()
    # 翻轉影像，參數 1 為水平翻轉成為鏡像
    img = cv2.flip(img, 1)
    # 將彩色影像轉換為灰階，這是作為辨識使用，不是要顯示出來的
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 利用 Haar cascades 進行臉部檢測
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20)
    )
    # 畫出每一個檢測到的臉部範圍
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]


    # 顯示含有標出臉部的影像
    cv2.imshow("video", img)
    # 檢查是否有按下 'ESC' 或 'q' 鍵
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or \
            cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1:
        break
        cv2.imshow("video", img)
    
    # 導入自訂的模組取代原本的鍵盤監區塊
    # 檢查是否有按下 'ESC' 或 'q' 鍵
#    if should_exit("video"):
#        break



# 釋放攝像頭資源並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()