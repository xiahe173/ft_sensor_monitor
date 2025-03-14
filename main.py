from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton
import sys
import serial
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGroupBox,
    QPushButton,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QFrame,
    QWidget,
    QMessageBox,
    QAbstractButton,
)

from PySide6.QtCore import QThread, QTimer, QProcess

import pyqtgraph as pg
import os
import time
import csv
import datetime
import argparse
import platform
import subprocess
import pandas as pd

# Setup argument parser
parser = argparse.ArgumentParser(description="Oscilloscope Application")
parser.add_argument(
    "-p", "--serial-port", default="COM12", help="Arduino serial port (default: COM12)"
)
parser.add_argument(
    "-b",
    "--baud-rate",
    type=int,
    default=115200,
    help="Baud rate for serial communication (default: 115200)",
)
parser.add_argument(
    "-n", "--num-channels", type=int, default=6, help="Number of channels (default: 6)"
)
parser.add_argument(
    "-f",
    "--b-prime-file",
    type=str,
    help="Path to the B' matrix Excel file",
    default="B_prime_custom.xlsx",
)

# Parse arguments
args = parser.parse_args()

# Use arguments
SERIAL_PORT = args.serial_port
BAUD_RATE = args.baud_rate
NUM_OF_CHANNELS = args.num_channels
# SG channels line colors starting from SG1 to SG6.
SG_LINE_COLORS = ["brown", "blue", "black", "grey", "green", "y"]
# the order of the incoming serial data mapped to SG channels. the first
# serial data corresponds to SG6, the second to SG1, etc.
# SG_ORDER = [1, 5, 6, 4, 2, 3]
SG_ORDER = [1, 6, 5, 4, 3, 2]

# Load B' matrix from the Excel file
B_prime_path = (
    args.b_prime_file
)  # The path can be obtained from the command-line arguments
B_prime_df = pd.read_excel(B_prime_path, header=None)  # No header in the file

# Assuming the first row and first column are headers, we remove them
# and convert the rest to a NumPy array of type float
B_prime_matrix = B_prime_df.iloc[1:, 1:].to_numpy(dtype=float)


class ControlBox(QGroupBox):
    def __init__(self, controller, parent=None):
        super().__init__("控制面板", parent=parent)
        self.controller = controller

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Serial Print Rate Selection
        self.rate_label = QLabel("串口输出速率 (ms):")
        layout.addWidget(self.rate_label)

        self.combobox_rate = QComboBox()
        self.combobox_rate.addItems(["1", "2", "5", "10", "20", "50", "100"])
        layout.addWidget(self.combobox_rate)
        # Connect the new combobox to its action
        self.combobox_rate.currentTextChanged.connect(self.set_serial_rate)

        # Tare Button
        self.tare_button = QPushButton("置零")
        layout.addWidget(self.tare_button)
        # Connect the tare button to its action
        self.tare_button.clicked.connect(self.send_tare_command)

        # 添加切换按钮
        self.toggle_plot_button = QPushButton("切换视图")
        layout.addWidget(self.toggle_plot_button)
        self.toggle_plot_button.clicked.connect(self.controller.toggle_plot_mode)

    def set_serial_rate(self, rate):
        command = f"s{rate}"
        # Assuming self.controller.serial_thread.serial_port is an open serial.Serial object
        if self.controller.serial_thread.serial_port.is_open:
            self.controller.serial_thread.serial_port.write(command.encode())
            self.controller.update_x_axis_based_on_rate(
                int(rate)
            )  # Update the X-axis based on the new rate

    def send_tare_command(self):
        command = "t"
        # Send the tare command over serial
        if self.controller.serial_thread.serial_port.is_open:
            self.controller.serial_thread.serial_port.write(command.encode())
            print("Tare command sent.")


class ControlPanel(QFrame):
    def __init__(self, controller, parent=None):
        super().__init__(parent=parent)
        self.controller = controller

        self.setFrameStyle(QFrame.StyledPanel)
        self.motor_control_panel = ControlBox(
            self.controller
        )  # Add motor control panel
        self.layout = QVBoxLayout()
        self.layout.addStretch()

        self.layout.addWidget(
            self.motor_control_panel
        )  # Include motor control panel in layout

        self.setLayout(self.layout)


class SerialReaderThread(QThread):
    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=0.1)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        self.calibration_values = [0.0] * 6  # Calibration values for each channel (reordered)

        self.data_buffer = []  # Shared buffer
        # Timing for interval measurement
        self.last_update_time = time.time()  # Initialize the last update time

    def set_calibration_values(self, calibration_values):
        self.calibration_values = calibration_values

    def run(self):
        interval_sum = 0.0  # Sum of intervals measured
        interval_count = 0  # Count of intervals measured
        last_print_time = time.time()  # Last time the average was printed

        while True:
            if self.serial_port.inWaiting() > 0:
                current_time = time.time()
                actual_interval = (current_time - self.last_update_time) * 1000  # ms
                self.last_update_time = current_time

                # Accumulate the sum and count of intervals
                interval_sum += actual_interval
                interval_count += 1

                try:
                    line = self.serial_port.readline().decode().strip()
                    if line:  # If the line is not empty
                        try:
                            # Attempt to parse the line as a list of floats
                            sensor_values = [float(val) for val in line.split(",")[:6]]

                            # Initialize a list with the same length as sensor_values filled
                            # with zeros
                            reordered_values = [0] * len(sensor_values)

                            # Reorder the sensor_values according to SG_ORDER
                            # SG_ORDER is treated as the target position for each corresponding
                            # value
                            for i, order in enumerate(SG_ORDER):
                                reordered_values[order - 1] = sensor_values[i]

                            # Apply calibration values to the reordered values
                            reordered_values = [
                                reordered_values[i] -
                                self.calibration_values[i] for i in range(6)]

                            self.data_buffer.append(reordered_values)
                        except ValueError as e:
                            # If conversion fails, it's not a data line
                            print("Message received:", line)
                except ValueError:
                    # This catches errors not related to parsing floats,
                    # e.g., if the line could not be decoded from UTF-8
                    print("Failed to decode serial data")

                # Check if a second has passed since the last print
                if current_time - last_print_time >= 1.0:
                    if interval_count > 0:
                        average_interval = interval_sum / interval_count
                        print(f"Average serial interval: {average_interval:.2f} ms")
                    else:
                        print("No data received in the last second.")

                    # Reset for the next second
                    last_print_time = current_time
                    interval_sum = 0.0
                    interval_count = 0


class Oscilloscope(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Oscilloscope, self).__init__(*args, **kwargs)
        self.plot_mode = "SG"  # 初始模式设为显示SG读数

        # Set the window title here
        self.setWindowTitle("6维度力传感器数据采集")
        # Main widget and layout
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        layout = QVBoxLayout(mainWidget)

        # Include the graph widget
        self.graphWidget = pg.PlotWidget()
        layout.addWidget(self.graphWidget)

        # Add the control panel
        self.controlPanel = ControlPanel(self)  # Pass self as the controller
        layout.addWidget(self.controlPanel)

        # Add a Save Data button
        self.save_data_button = QPushButton("保存绘图数据")
        layout.addWidget(self.save_data_button)
        # Connect the button to the save method
        self.save_data_button.clicked.connect(self.save_plot_data)

        # Configuration for 10 seconds window with 100ms interval
        self.time_window = 10  # seconds
        interval = 0.02  # serial port transmission rate interval in seconds
        num_points = int(self.time_window / interval)

        # Initialize channel data storage
        self.channel_data = [np.zeros(num_points) for _ in range(NUM_OF_CHANNELS)]

        # Continue with the rest of your initialization
        self.colors = SG_LINE_COLORS
        self.number_of_channels = NUM_OF_CHANNELS

        self.redraw_graph_for_current_mode()
        # Oscilloscope settings
        self.x = np.linspace(0, self.time_window, num_points)
        self.y_sensor = np.zeros(num_points)  # Data points for sensor

        self.graphWidget.setBackground("w")
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setXRange(0, self.time_window, padding=0.02)
        self.graphWidget.setYRange(0, 5, padding=0.02)
        self.graphWidget.setLabel("left", "数值")
        self.graphWidget.setLabel("bottom", "时间 (s)")

        self.serial_thread = SerialReaderThread(SERIAL_PORT, BAUD_RATE)
        self.serial_thread.start()

        self.timer = QTimer()
        # Refresh rate is twice the serial port transmission rate
        fixed_ui_interval = 0.02
        self.timer.setInterval(int(fixed_ui_interval * 1000))  # in milliseconds
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def redraw_graph_for_current_mode(self):
        # Clear the existing graph
        self.graphWidget.clear()
        self.legend = self.graphWidget.addLegend()
        self.legend.anchor(
            (0, 0), (0, 0)
        )  # Anchor points for legend and plot respectively
        # Determine which mode we're in and draw accordingly
        if self.plot_mode == "SG":
            self.graphWidget.setTitle("6通道传感器度数", color='blue', size='20pt')
            labels = ["SG1", "SG2", "SG3", "SG4", "SG5", "SG6"]
            self.data_lines = [
                self.graphWidget.plot(
                    [], [], pen=pg.mkPen(color=self.colors[i], width=2), name=label
                )
                for i, label in enumerate(labels)
            ]
        else:  # "Forces" mode
            self.graphWidget.setTitle("6维力计算值", color='red', size='20pt')
            labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
            self.data_lines = [
                self.graphWidget.plot(
                    [],
                    [],
                    pen=pg.mkPen(color=self.colors[i % len(self.colors)], width=2),
                    name=label,
                )
                for i, label in enumerate(labels)
            ]

    def toggle_plot_mode(self):
        # Toggle between sensor data mode and force data mode
        if self.plot_mode == "SG":
            self.plot_mode = "Forces"
        else:
            self.plot_mode = "SG"
        # Redraw the graph for the new mode
        self.redraw_graph_for_current_mode()

    def calculate_forces(self, channel_data_history):
        # Stack the channel data history to form a 2D array with shape (6, N)
        # Where N is the number of time points
        sg_values_matrix = np.vstack(channel_data_history)  # shape (6, N)

        # Calculate the forces for all time points using B_prime_matrix
        # The resulting forces_matrix will have shape (6, N)
        forces_matrix = np.dot(B_prime_matrix, sg_values_matrix)

        return forces_matrix

    def update_plot_with_forces(self, forces_matrix):
        # Update the plot lines with the calculated forces
        for i in range(self.number_of_channels):
            # Update the plot with new forces data without modifying channel_data
            # forces_matrix[i, :] should be a 1D array with the same length as self.x
            self.data_lines[i].setData(self.x, forces_matrix[i, :])

    def update_plot_with_sensors(self, sensor_data_buffer):
        # Update the plot lines with sensor data
        for i in range(self.number_of_channels):
            # sensor_data_buffer[i] should be a 1D array with the same length as self.x
            self.data_lines[i].setData(self.x, sensor_data_buffer[i])

    def adjust_y_axis(self, minY, maxY):
        margin = (maxY - minY) * 0.1  # 10% margin
        if minY == maxY:
            minY -= 0.5
            maxY += 0.5
        self.graphWidget.setYRange(minY - margin, maxY + margin)

    def update_plot_data(self):
        # Get the latest sensor data
        if self.serial_thread.data_buffer:
            # Copy and clear the buffer
            data = self.serial_thread.data_buffer[:]
            self.serial_thread.data_buffer.clear()

            # Process new sensor data and update the channel_data history
            for i, readings in enumerate(zip(*data)):
                num_new_values = len(readings)
                self.channel_data[i] = np.roll(self.channel_data[i], -num_new_values)
                self.channel_data[i][-num_new_values:] = readings

            if self.plot_mode == "Forces":
                # Calculate forces from sensor data
                forces_matrix = self.calculate_forces(self.channel_data)
                # Update the plot with force data
                self.update_plot_with_forces(forces_matrix)
                # Adjust Y-axis based on the forces data
                self.adjust_y_axis_based_on_data(forces_matrix)
            else:
                # Update the plot with sensor data
                self.update_plot_with_sensors(self.channel_data)
                # Adjust Y-axis based on the sensor data
                self.adjust_y_axis_based_on_data(self.channel_data)

    def adjust_y_axis_based_on_data(self, y_data):
        # Assuming y_data is a list of arrays, one for each channel
        all_y_data = np.concatenate(y_data)
        minY, maxY = np.min(all_y_data), np.max(all_y_data)
        margin = (maxY - minY) * 0.1  # 10% margin
        if minY == maxY:
            minY -= 0.5  # Ensuring there is always a range to display
            maxY += 0.5
        self.graphWidget.setYRange(minY - margin, maxY + margin)

    def update_x_axis_based_on_rate(self, rate_ms):
        self.current_serial_rate_ms = rate_ms
        interval = rate_ms / 1000.0  # Convert ms to seconds
        num_points = int(self.time_window / interval)
        self.x = np.linspace(0, self.time_window, num_points)

        # Resize each channel's data array to match the new x array length
        for i in range(len(self.channel_data)):
            current_length = len(self.channel_data[i])
            if current_length < num_points:
                # If the current data array is shorter, extend it with zeros
                self.channel_data[i] = np.append(
                    self.channel_data[i], np.zeros(num_points - current_length)
                )
            else:
                # If it's longer, truncate it to match the new length
                self.channel_data[i] = self.channel_data[i][:num_points]

    def save_plot_data(self):
        # Your existing code to save data to a CSV file...

        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"record_{timestamp}.csv"

        # Open a file with newline='' so that the newline character in the file is
        # controlled by the csv writer
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Prepare header with channel names
            headers = ["Time"] + [f"CH{i+1}" for i in range(self.number_of_channels)]
            writer.writerow(headers)  # Write the column headers

            # Assuming self.x is common for all channels and represents the time
            for i in range(len(self.x)):
                row = [self.x[i]]  # Start with time
                for channel_data in self.channel_data:
                    row.append(channel_data[i])  # Add data point from each channel
                writer.writerow(row)  # Write the row to the CSV

        # Generate a message box with custom actions
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(f"绘图数据已保存到：{filename}")
        msgBox.setWindowTitle("已保存")

        # Standard button for OK
        okButton = msgBox.addButton(QMessageBox.StandardButton.Ok)
        # Custom buttons for additional actions
        showInExplorerButton = msgBox.addButton(
            "文件浏览器中显示", QMessageBox.ActionRole
        )
        openFileButton = msgBox.addButton("打开文件", QMessageBox.ActionRole)

        msgBox.exec()  # Display the message box

        # Check which button was clicked
        clickedButton = msgBox.clickedButton()
        if clickedButton == showInExplorerButton:
            self.show_in_explorer(filename)
        elif clickedButton == openFileButton:
            self.open_file(filename)

    def show_in_explorer(self, filename):
        full_path = os.path.abspath(filename)
        if platform.system() == "Windows":
            # For Windows, use explorer with /select, to highlight the file
            subprocess.run(f'explorer /select,"{full_path}"', shell=True)
        elif platform.system() == "Darwin":  # macOS
            # macOS doesn't support selecting a file via open command, so just open the folder
            folder = os.path.dirname(full_path)
            subprocess.run(["open", folder], check=True)
        else:  # Linux
            # For Linux, the behavior can vary between file managers; the following is for Nautilus (GNOME)
            # Attempt to use `gio` as a more universal approach where available
            try:
                subprocess.run(["gio", "open", full_path], check=True)
            except subprocess.CalledProcessError:
                # Fallback to opening the folder if gio open fails or is not available
                folder = os.path.dirname(full_path)
                subprocess.run(["xdg-open", folder], check=True)

    def open_file(self, filename):
        # Open the file with the default application
        if platform.system() == "Windows":
            os.startfile(filename)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", filename])
        else:  # Linux variants
            subprocess.Popen(["xdg-open", filename])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    oscilloscope = Oscilloscope()
    oscilloscope.show()
    sys.exit(app.exec())
