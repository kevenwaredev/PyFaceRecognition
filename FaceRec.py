import cv2
import mediapipe as mp

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Função para desenhar landmarks
def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    for i in range(len(landmarks) - 1):
        cv2.line(image, tuple(landmarks[i]), tuple(landmarks[i + 1]), (255, 0, 0), 1)
    return image

# Função para obter landmarks faciais de uma imagem
def get_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                landmarks.append((x, y))
        return landmarks
    return None

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_landmarks(frame)
    if landmarks is not None:
        frame = draw_landmarks(frame, landmarks)

    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
