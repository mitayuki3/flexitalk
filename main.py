import sys
import os
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QFileSystemWatcher, QObject, Slot, QUrl

class HotReloader(QObject):
    def __init__(self, engine, qml_file):
        super().__init__()
        self.engine = engine
        self.qml_file = qml_file
        self.watcher = QFileSystemWatcher([qml_file])
        self.watcher.fileChanged.connect(self.reload)

    @Slot(str)
    def reload(self, path):
        print(f"File changed: {path}. Reloading...")
        
        # すべてのトップレベルウィンドウを閉じる
        for obj in self.engine.rootObjects():
            obj.deleteLater()
            
        # キャッシュをクリア
        self.engine.clearComponentCache()
        
        # 再ロード
        self.engine.load(QUrl.fromLocalFile(self.qml_file))
        
        if not self.engine.rootObjects():
            print("Reload failed: no root objects.")

def main():
    app = QGuiApplication(sys.argv)
    
    engine = QQmlApplicationEngine()
    qml_path = os.path.join(os.path.dirname(__file__), "Main.qml")
    
    # 初回ロード
    engine.load(QUrl.fromLocalFile(qml_path))
    
    if not engine.rootObjects():
        sys.exit(-1)
        
    # ホットリローダーの初期化
    reloader = HotReloader(engine, qml_path)
    
    # reloaderがガベージコレクションされないように参照を保持
    app.reloader = reloader
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
