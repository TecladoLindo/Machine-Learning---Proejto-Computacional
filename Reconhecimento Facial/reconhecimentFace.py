import cv2
import mediapipe as mp


def FacialRetangulo():
    video_capture = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
       )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        eyes = eyeCascade.detectMultiScale(gray, 1.2, 18)
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        smiles = smileCascade.detectMultiScale(gray, 1.7, 20)
        for (x, y, w, h) in smiles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def FacialLinhas():
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

def FacialPontos():
    webcam = cv2.VideoCapture(0)

    #Utilizar o mediapipe para mostrar o reconhecimento
    solucao_reconhecimento_rosto = mp.solutions.face_detection
    reconhecedor_rosto = solucao_reconhecimento_rosto.FaceDetection()
    desenho = mp.solutions.drawing_utils

    while True: #cria um loop
        #Ler as informações da webcam:
        verificador, frame = webcam.read()
    
        if not verificador:
            break

        # reconhecer os rostos que tem ali dentro:
        lista_rostos = reconhecedor_rosto.process(frame)

        # desenhar os rostos na imagem:
        if lista_rostos.detections: 
            for rosto in lista_rostos.detections:
                desenho.draw_detection(frame, rosto)
    
        cv2.imshow("Rostos na Webcam", frame)
   
        # quando apertar ESC, para o loop:
        if cv2.waitKey(5) == ord('q'): # esse 27 é o código da tecla
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    FacialLinhas()