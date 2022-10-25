import cv2
import face_recognition

castiel1 = cv2.imread("./img/castiel1.jpg")
rgb_roger1 = cv2.cvtColor(castiel1, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_roger1)[0]

castiel2 = cv2.imread("./img/castiel2.jpg")
rgb_romulo1 = cv2.cvtColor(castiel2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_romulo1)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Castiel 1", castiel1)
cv2.imshow("Castiel 2", castiel2)
cv2.waitKey(0)