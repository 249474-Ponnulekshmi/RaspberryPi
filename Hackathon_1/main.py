import sys
from PyQt5 import QtWidgets
from logic import MainApp

if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets
    from logic import MainApp

    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())