# O CÓDIGO NÃO ESTÁ RODANDO E EU N SEI PQ
import cv2
import mediapipe as mp

# Inicializa captura de vídeo e o MediaPipe Face Mesh
video_capture = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Inicializa o desenho das malhas faciais
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    ret, frame = video_capture.read()

    # Converte a imagem para RGB (necessário para o MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem e obtém as malhas faciais
    results = face_mesh.process(rgb_frame)

    # Se forem detectadas faces, desenha os pontos
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Exibe o vídeo com as detecções
    cv2.imshow('video', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
video_capture.release()
cv2.destroyAllWindows()
# O CÓDIGO NÃO ESTÁ RODANDO E EU N SEI PQ
