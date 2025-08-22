import serial
import serial.tools.list_ports
import time
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from pymodbus.client import ModbusSerialClient as ModbusClient

class UartSerial(QObject):

    def __init__(self, port='COM11', baud_rate=1000000):
        self.ser = None
        self.port = port
        self.baud_rate = baud_rate

    # # 清空缓冲区
    # def clear_recv_buffer(self):
    #     # 清空串口接收缓冲区。
    #     while self.client.serial.in_waiting() > 0:
    #         self.client.serial.read(self.client.serial.in_waiting())

    # 检测串口是否打开
    def is_port_open(self):
        try:
            # 检查 Modbus 客户端是否已创建并连接
            if self.ser.is_open:
                print("串口已经打开")
                return False
            else:
                print("串口未打开")
                return True
        except Exception as e:
            print(f"检查串口时发生错误: {e}")
            return True

    # 串口检测
    def get_all_port(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.port_list_name = []
        port_list = list(serial.tools.list_ports.comports())
        i = 0

        if len(port_list) <= 0:
            return []
        else:
            for port in port_list:
                i = i + 1
                self.port_list_name.append(port[0])

        return self.port_list_name

    # 打开串口(TXZ)
    def OpenSerialPort(self):
        self.ser = serial.Serial(self.port, self.baud_rate)
        if self.ser.is_open:
            print('串口打开成功')
        else:
            print('串口打开失败')

    # 关闭串口(TXZ)
    def CloseSerialPort(self):
        self.ser.close()
        if self.ser.is_open:
            print('串口未关闭')
        else:
            print('串口已关闭')

    # 尝试打开串口
    def try_port_open(self, port_name, baud_rate):
        self.port = port_name
        self.baud_rate = baud_rate
        try:
            self.OpenSerialPort()
            return True
        except Exception as e:
            print(f"打开串口发生错误: {e}")
            return False

    # 串口写(TXZ)
    def Write(self, message):
        self.ser.write(message)

    # 串口读(TXZ)
    def Read(self, length):
        return self.ser.read(length)

    # 驱动器使能(TXZ)
    def Enable(self, id, enable):
         self.Write(self.CalTx(0x00, enable, id, 0x06))
         if 0 < id < 8:
              RX = self.Read(8)
              if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                   if RX[:6] == bytes([id, 0x06, 0x00, 0x00, 0x00, enable]):
                        print('Enable')
                        return True
              return False

    # 设置驱动器PWM(TXZ)
    def SetPWM(self, id, pwm):
         self.Write(self.CalTx(0x01, pwm, id, 0x06))
         if 0 < id < 8:
              RX = self.Read(8)
              if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                   expected_response = bytes([id, 0x06, 0x00, 0x01, (pwm >> 8) & 0xFF, pwm & 0xFF])
                   if RX[:6] == expected_response:
                        # print('SetPWM')
                        return True
              return False

    # 应用设置的PWM(TXZ)
    def AppPWM(self, id):
         self.Write(self.CalTx(0x02, 0x01, id, 0x06))
         if 0 < id < 8:
              RX = self.Read(8)
              if self.Crc16_MODBUS(RX[:6]) == int.from_bytes(RX[6:], byteorder='little'):
                   if RX[:6] == bytes([id, 0x06, 0x00, 0x02, 0x00, 0x01]):
                        # print('Apply PWM')
                        return True
              return False

    # 读电流(TXZ)
    def ReadCurrent(self, id):
         if 0 < id < 8:
              self.Write(self.CalTx(0x05, 1, id, 0x03))
              RX = self.Read(7)
              if self.Crc16_MODBUS(RX[:5]) == int.from_bytes(RX[5:], byteorder='little'):
                   current_raw = (RX[3] << 8) | RX[4]
                   if current_raw >= 0x8000:
                        current = current_raw - 0x10000
                   else:
                        current = current_raw
                   return current
              return False
         else:
              print('WrongID')
              return False

    # 给Modbus计算校验位(TXZ)
    def Crc16_MODBUS(self,datas):
        crc = 0xFFFF
        poly = 0xA001
        for byte in datas:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
        return crc

    # 按包的格式做封装(TXZ)
    def CalTx(self, register_address, value, id, function_code):
        modbus_frame = [
            id,
            function_code,
            (register_address >> 8) & 0xFF,
            register_address & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF
        ]
        crc = self.Crc16_MODBUS(modbus_frame)
        modbus_frame.append(crc & 0xFF)
        modbus_frame.append((crc >> 8) & 0xFF)
        return bytes(modbus_frame)


# a = UartSerial()
# a.is_port_open()
# print(a.get_all_port())
# a.OpenSerialPort()
# a.Enable(0,0)