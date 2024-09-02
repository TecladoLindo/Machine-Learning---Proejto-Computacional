
import sys 
from PyQt6.QtWidgets import *
import reconhecimentFace as rf

class Janela(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Layout")
        self.criar_widgets()

    def criar_widgets(self):
        fp = QPushButton("Facial pontos")
        fp.clicked.connect(self.clicked_fp)

        fl = QPushButton("Facial linhas")
        fl.clicked.connect(self.clicked_fl)

        fr = QPushButton("Facial retangulo")
        fr.clicked.connect(self.clicked_fr)

        grid = QGridLayout()
        grid.addWidget(fp, 0,0)
        grid.addWidget(fl, 0,1)
        grid.addWidget(fr, 0,2)
        self.setLayout(grid) 
        
    def clicked_fp(self):
        rf.FacialPontos()
    
    def clicked_fl(self):
        rf.FacialLinhas()
    
    def clicked_fr(self):
        rf.FacialRetangulo()

app = QApplication(sys.argv)
window = Janela()
window.show()
sys.exit(app.exec())
        
