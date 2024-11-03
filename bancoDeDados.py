import sqlite3
import pandas as pd
import time
from datetime import datetime

#conecta ao bando de dados sqlite3
conexao = sqlite3.connect("reconhecimento_facial.db")
cursor = conexao.cursor()

#tabela de reconhecimento facial
cursor.execute('''
CREATE TABLE IF NOT EXISTS reconhecimento_facial (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    tempo_deteccao REAL NOT NULL,
    data_hora TEXT NOT NULL,
    landmark_0_x REAL, landmark_0_y REAL, landmark_0_z REAL,
    landmark_1_x REAL, landmark_1_y REAL, landmark_1_z REAL,
    landmark_2_x REAL, landmark_2_y REAL, landmark_2_z REAL,
    landmark_3_x REAL, landmark_3_y REAL, landmark_3_z REAL,
    landmark_4_x REAL, landmark_4_y REAL, landmark_4_z REAL,
    landmark_5_x REAL, landmark_5_y REAL, landmark_5_z REAL,
    landmark_6_x REAL, landmark_6_y REAL, landmark_6_z REAL,
    landmark_7_x REAL, landmark_7_y REAL, landmark_7_z REAL,
    landmark_8_x REAL, landmark_8_y REAL, landmark_8_z REAL,
    landmark_9_x REAL, landmark_9_y REAL, landmark_9_z REAL,
    landmark_10_x REAL, landmark_10_y REAL, landmark_10_z REAL,
    landmark_11_x REAL, landmark_11_y REAL, landmark_11_z REAL,
    landmark_12_x REAL, landmark_12_y REAL, landmark_12_z REAL
)
''')
conexao.commit()

#funcao pra registrar dados de reconhecimento facial
def registrar_reconhecimento(nome, landmarks):
    inicio = time.time()
    time.sleep(2)  #tempo de detecção
    tempo_deteccao = time.time() - inicio #cálculo do tempo de detecção
    data_hora = datetime.now().isoformat()

    df_landmarks = pd.DataFrame(landmarks, columns=['x', 'y', 'z']) #transforma landmarcks em dataframes do pandas
    valores_landmarks = df_landmarks.values.flatten().tolist() #lista de valores dos landmarks
    
    # Inserir os dados no banco de dados
    cursor.execute('''
        INSERT INTO reconhecimento_facial (
            nome, tempo_deteccao, data_hora, 
            landmark_0_x, landmark_0_y, landmark_0_z,
            landmark_1_x, landmark_1_y, landmark_1_z,
            landmark_2_x, landmark_2_y, landmark_2_z,
            landmark_3_x, landmark_3_y, landmark_3_z,
            landmark_4_x, landmark_4_y, landmark_4_z,
            landmark_5_x, landmark_5_y, landmark_5_z,
            landmark_6_x, landmark_6_y, landmark_6_z,
            landmark_7_x, landmark_7_y, landmark_7_z,
            landmark_8_x, landmark_8_y, landmark_8_z,
            landmark_9_x, landmark_9_y, landmark_9_z,
            landmark_10_x, landmark_10_y, landmark_10_z,
            landmark_11_x, landmark_11_y, landmark_11_z,
            landmark_12_x, landmark_12_y, landmark_12_z
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', [nome, tempo_deteccao, data_hora] + valores_landmarks)
    
    conexao.commit()
    print(f"Dados de reconhecimento facial para {nome} registrados com sucesso.")

#exemplos de landmarcks
landmarks_exemplo = [
    {"x": 0.5, "y": 0.5, "z": 0.1},
    {"x": 0.6, "y": 0.6, "z": 0.2},
    {"x": 0.7, "y": 0.7, "z": 0.3},
    {"x": 0.8, "y": 0.8, "z": 0.4},
    {"x": 0.9, "y": 0.9, "z": 0.5},
    {"x": 1.0, "y": 1.0, "z": 0.6},
    {"x": 1.1, "y": 1.1, "z": 0.7},
    {"x": 1.2, "y": 1.2, "z": 0.8},
    {"x": 1.3, "y": 1.3, "z": 0.9},
    {"x": 1.4, "y": 1.4, "z": 1.0},
    {"x": 1.5, "y": 1.5, "z": 1.1},
    {"x": 1.6, "y": 1.6, "z": 1.2},
    {"x": 1.7, "y": 1.7, "z": 1.3},
]

#registrar usuário
registrar_reconhecimento("João Silva", landmarks_exemplo)


conexao.close()
