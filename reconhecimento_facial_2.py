import cv2 
import mediapipe as mp

#conectar o opencv na webcam
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
