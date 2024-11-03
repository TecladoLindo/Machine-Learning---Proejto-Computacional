import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

mp_face_mesh = mp.solutions.face_mesh

def selecionar_landmarks_principais(landmarks):
    pontos_principais = [33, 133, 362, 263, 1, 9, 10, 152, 168, 234, 454, 323, 93]  
    landmarks_principais = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in pontos_principais]
    return landmarks_principais

def calcular_distancia_normalizada(landmarks):
    distancias = []
    for i in range(len(landmarks) - 1):
        for j in range(i + 1, len(landmarks)):
            distancias.append(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j])))
    return distancias

def calcular_diferenca_landmarks(landmarks1, landmarks2):
    distancias1 = calcular_distancia_normalizada(landmarks1)
    distancias2 = calcular_distancia_normalizada(landmarks2)
    diferenca = np.abs(np.array(distancias1) - np.array(distancias2))
    return np.mean(diferenca)

def capturar_rosto_autorizado():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            success, image = cap.read()
            if not success:
                print("Falha ao capturar a imagem.")
                break
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_principais = selecionar_landmarks_principais(face_landmarks.landmark)
                    with open('rosto_autorizado.pkl', 'wb') as f:
                        pickle.dump(landmarks_principais, f)
                    print("Rosto autorizado capturado e salvo.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cv2.imshow('Capturar Rosto Autorizado', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def ajustar_tolerancia_por_distancia(landmarks_atual):
    distancias = calcular_distancia_normalizada(landmarks_atual)
    distancia_media = np.mean(distancias)
    if distancia_media < 0.015:  
        return 0.15  
    elif distancia_media > 0.05:  
        return 0.2  
    else:
        return 0.10  

def verificar_rosto_autorizado():
    try:
        with open('rosto_autorizado.pkl', 'rb') as f:
            rosto_autorizado = pickle.load(f)
    except FileNotFoundError:
        print("Nenhum rosto autorizado foi salvo. Por favor, capture o rosto primeiro.")
        return False
    
    cap = cv2.VideoCapture(0)
    autorizacoes_consecutivas = 0  
    frame_check_threshold = 3  
    tempo_limite = 10  
    inicio_verificacao = time.time()  

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            tempo_decorrido = time.time() - inicio_verificacao
            if tempo_decorrido > tempo_limite:
                print("Tempo limite excedido. Verificação facial falhou.")
                break

            success, image = cap.read()
            if not success:
                print("Falha ao capturar a imagem.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_atual = selecionar_landmarks_principais(face_landmarks.landmark)
                    limite_diferenca = ajustar_tolerancia_por_distancia(landmarks_atual)
                    diferenca = calcular_diferenca_landmarks(landmarks_atual, rosto_autorizado)
                    
                    if diferenca < limite_diferenca:  
                        autorizacoes_consecutivas += 1
                        print(f"Rosto autorizado detectado! (Contagem: {autorizacoes_consecutivas})")
                        
                        if autorizacoes_consecutivas >= frame_check_threshold:
                            print("Dispositivo desbloqueado.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return True
                    else:
                        autorizacoes_consecutivas = 0
                        print(f"Rosto não autorizado. Diferença: {diferenca:.4f}")
                    
            cv2.imshow('Verificação Facial', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return False

def menu():
    while True:
        print("\nMenu:")
        print("1. Capturar rosto autorizado")
        print("2. Verificar rosto")
        print("3. Sair")
        opcao = input("Escolha uma opção: ")
        
        if opcao == '1':
            capturar_rosto_autorizado()
        elif opcao == '2':
            if verificar_rosto_autorizado():
                print("Dispositivo desbloqueado.")
            else:
                print("Dispositivo bloqueado.")
        elif opcao == '3':
            break
        else:
            print("Opção inválida, tente novamente.")
menu()
