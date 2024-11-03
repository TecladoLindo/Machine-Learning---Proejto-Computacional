import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Função para selecionar landmarks principais (exemplo: olhos, nariz, boca)
def selecionar_landmarks_principais(landmarks):
    # IDs dos principais pontos nodais: olhos, nariz e boca
    pontos_principais = [33, 133, 362, 263, 1, 9, 10, 152, 168, 234, 454, 323, 93]  # IDs dos landmarks mais estáveis
    landmarks_principais = [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in pontos_principais]
    return landmarks_principais

# Função para calcular as distâncias normalizadas entre os landmarks
def calcular_distancia_normalizada(landmarks):
    distancias = []
    # Calcula as distâncias normalizadas entre os landmarks principais
    for i in range(len(landmarks) - 1):
        for j in range(i + 1, len(landmarks)):
            distancias.append(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j])))
    return distancias

# Função para calcular a diferença média entre as distâncias normalizadas de dois conjuntos de landmarks
def calcular_diferenca_landmarks(landmarks1, landmarks2):
    distancias1 = calcular_distancia_normalizada(landmarks1)
    distancias2 = calcular_distancia_normalizada(landmarks2)
    diferenca = np.abs(np.array(distancias1) - np.array(distancias2))
    return np.mean(diferenca)

# Função para capturar e salvar os landmarks do rosto autorizado
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
                    # Extrai landmarks principais
                    landmarks_principais = selecionar_landmarks_principais(face_landmarks.landmark)
                    
                    # Salva os landmarks principais do rosto autorizado em um arquivo
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

# Função para ajustar a tolerância de acordo com a distância do rosto à câmera
def ajustar_tolerancia_por_distancia(landmarks_atual):
    # Calcula uma média das distâncias entre os principais landmarks
    distancias = calcular_distancia_normalizada(landmarks_atual)
    distancia_media = np.mean(distancias)

    # Define um limite de diferença ainda mais flexível para rostos mais distantes
    if distancia_media < 0.015:  # Rosto muito próximo
        return 0.15  # Tolerância maior para curtas distâncias
    elif distancia_media > 0.05:  # Rosto muito distante
        return 0.2  # Tolerância bem maior para longas distâncias
    else:
        return 0.10  # Tolerância intermediária para distâncias normais

# Função para verificar se o rosto atual corresponde ao rosto autorizado
def verificar_rosto_autorizado():
    try:
        # Carrega os landmarks do rosto autorizado
        with open('rosto_autorizado.pkl', 'rb') as f:
            rosto_autorizado = pickle.load(f)
    except FileNotFoundError:
        print("Nenhum rosto autorizado foi salvo. Por favor, capture o rosto primeiro.")
        return False
    
    cap = cv2.VideoCapture(0)
    autorizacoes_consecutivas = 0  # Contador para múltiplas autorizações consecutivas
    frame_check_threshold = 3  # Exigir 3 frames consecutivos autorizados para desbloquear
    tempo_limite = 10  # Limite de tempo para verificação (em segundos)
    inicio_verificacao = time.time()  # Marca o início da verificação

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while True:
            # Verifica se o tempo limite foi excedido
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
                    # Extrai landmarks principais do rosto atual
                    landmarks_atual = selecionar_landmarks_principais(face_landmarks.landmark)
                    
                    # Ajusta a tolerância dinamicamente com base na distância do rosto
                    limite_diferenca = ajustar_tolerancia_por_distancia(landmarks_atual)
                    
                    # Calcula a diferença média entre o rosto atual e o autorizado
                    diferenca = calcular_diferenca_landmarks(landmarks_atual, rosto_autorizado)
                    
                    if diferenca < limite_diferenca:  # Tolerância ajustada dinamicamente
                        autorizacoes_consecutivas += 1
                        print(f"Rosto autorizado detectado! (Contagem: {autorizacoes_consecutivas})")
                        
                        # Se houver autorizações suficientes, desbloqueia
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

# Menu para selecionar ação
def menu():
    while True:
        print("\nMenu:")
        print("1. Capturar rosto autorizado")
        print("2. Verificar rosto para desbloquear")
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

# Executa o menu
menu()
