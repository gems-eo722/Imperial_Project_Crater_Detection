import sys
from collections import OrderedDict

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QStackedWidget, QPushButton, \
    QGridLayout, QVBoxLayout, QComboBox, QSpacerItem, QMessageBox

import tycho_cdm.tycho
from tycho_cdm.model.TychoCDM import TychoCDM


class TychoGUI(QWidget):

    def __init__(self):
        super(TychoGUI, self).__init__()

        # Variables to extract from UI
        self.planet_name = None
        self.output_folder_path = None
        self.input_folder_path = None

        # Main page elements
        self.batch_mode_button = QPushButton("Batch mode\n(Specify input folder)")
        self.single_mode_button = QPushButton("Single image mode")

        # Batch page elements
        self.output_folder_path_text = None
        self.input_folder_path_text = None
        self.planet_selection_dropdown = None
        self.batch_submit_button = None

        # Define pages
        self.page_1 = QWidget()
        self.page_2 = QWidget()
        self.page_3 = QWidget()

        # Fill pages
        self.fill_page_1()
        self.fill_page_2()
        self.fill_page_3()

        # Put pages on stacked widget
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.page_1)
        self.stacked_widget.addWidget(self.page_2)

        # Global layout, allows cancel button to be separate from page-stack
        hbox = QVBoxLayout(self)
        hbox.addWidget(self.stacked_widget)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        hbox.addWidget(self.cancel_button)

        # Button callbacks
        self.cancel_button.clicked.connect(self.cancel)
        self.batch_mode_button.clicked.connect(
            lambda: (self.stacked_widget.setCurrentIndex(1),
                     self.cancel_button.setVisible(True)))
        self.single_mode_button.clicked.connect(
            lambda: (self.stacked_widget.setCurrentIndex(2),
                     self.cancel_button.setVisible(True)))

        # Final settings
        self.setLayout(hbox)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Tycho CDM')
        self.show()

    def fill_page_1(self):
        layout = QGridLayout()
        layout.addWidget(self.batch_mode_button)
        layout.addWidget(self.single_mode_button)
        self.page_1.setLayout(layout)

    def fill_page_2(self):
        layout = QGridLayout()

        input_select_button = QPushButton("Select input folder")
        input_select_button.clicked.connect(
            lambda: self.choose_file_or_folder_to_field("input_folder_path", self.select_folder, "Select input folder"))
        self.input_folder_path_text = QLabel("")

        output_select_button = QPushButton("Select output folder")
        output_select_button.clicked.connect(
            lambda: self.choose_file_or_folder_to_field("output_folder_path", self.select_folder,
                                                        "Select output folder"))
        self.output_folder_path_text = QLabel("")

        planet_selection_layout = QVBoxLayout()
        planet_text = QLabel("Select planet")
        self.planet_selection_dropdown = QComboBox()
        items = OrderedDict([('None', ''), ('Mars', ''), ('Moon', '')])
        self.planet_selection_dropdown.addItems(items.keys())
        self.planet_selection_dropdown.setCurrentIndex(0)
        self.planet_selection_dropdown.currentTextChanged.connect(self.set_planet_name)
        planet_selection_layout.addWidget(planet_text)
        planet_selection_layout.addWidget(self.planet_selection_dropdown)

        self.batch_submit_button = QPushButton("Submit")
        self.batch_submit_button.clicked.connect(lambda: self.submit_batch())
        self.batch_submit_button.setEnabled(False)

        layout.addWidget(input_select_button, 0, 0)
        layout.addWidget(self.input_folder_path_text, 1, 0)

        layout.addWidget(output_select_button, 2, 0)
        layout.addWidget(self.output_folder_path_text, 3, 0)

        layout.addItem(QSpacerItem(0, 40), 4, 0)

        layout.addLayout(planet_selection_layout, 5, 0)

        layout.addItem(QSpacerItem(0, 40), 6, 0)

        layout.addWidget(self.batch_submit_button, 7, 0)

        self.page_2.setLayout(layout)

    def fill_page_3(self):
        pass

    def display(self, i):
        self.stacked_widget.setCurrentIndex(i)

    def cancel(self):
        self.stacked_widget.setCurrentIndex(0)
        self.cancel_button.setVisible(False)
        self.planet_selection_dropdown.setCurrentIndex(0)

        self.input_folder_path = None
        self.output_folder_path = None
        self.planet_name = None

        self.input_folder_path_text.setText("")
        self.output_folder_path_text.setText("")

    def is_batch_form_complete(self):
        return self.input_folder_path is not None \
            and self.output_folder_path is not None \
            and self.planet_name is not None

    def update_submit_button(self):
        if self.is_batch_form_complete():
            self.batch_submit_button.setEnabled(True)
        else:
            self.batch_submit_button.setEnabled(False)

    def choose_file_or_folder_to_field(self, field, method, window_name):
        setattr(self, field, method(window_name))  # Try setting field to path returned by File/Folder picker
        if getattr(self, field) == "":  # File/Folder picker did not return path
            setattr(self, field, None)  # Set path to None
            getattr(self, field + "_text").setText("")  # Set text for path to empty
        else:
            getattr(self, field + "_text").setText(getattr(self, field))
        self.update_submit_button()

    def set_planet_name(self, value):
        if value == "None":
            self.planet_name = None
        else:
            self.planet_name = value
        self.update_submit_button()

    def submit_batch(self):
        try:
            images_path, labels_path, data_path, planet_name, output_folder_path = \
                tycho_cdm.tycho.process_arguments(
                    self.input_folder_path, self.output_folder_path, self.planet_name)

            model = TychoCDM(planet_name)
            results = model.batch_inference(images_path)
            tycho_cdm.tycho.write_results(results, labels_path, data_path, output_folder_path)
            self.message_popup(f"Success! All results have been written to the output path:\n\n{output_folder_path}",
                               "Operation Complete", QMessageBox.Information)
        except RuntimeError as runtime_error:
            self.message_popup(runtime_error.args[0], "Error", QMessageBox.Critical)

    def message_popup(self, text, title, icon):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.exec_()

    def select_folder(self, window_name):
        dialog = QtWidgets.QFileDialog()
        result = dialog.getExistingDirectory(self, window_name)
        print(result)
        return result

    def select_file(self, window_name):
        dialog = QtWidgets.QFileDialog()
        result = dialog.getOpenFileName(self, window_name)[0]
        print(result)
        return result


def main():
    app = QApplication(sys.argv)
    _ = TychoGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
