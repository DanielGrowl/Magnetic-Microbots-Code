import serial
import serial.tools.list_ports
import time
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from pymodbus.client import ModbusSerialClient as ModbusClient

class UartSerial(QObject):

    def __init__(self):
        super(UartSerial, self).__init__()
        self.client = None

    # # 清空缓冲区
    # def clear_recv_buffer(self):
    #     # 清空串口接收缓冲区。
    #     while self.client.serial.in_waiting() > 0:
    #         self.client.serial.read(self.client.serial.in_waiting())

    def is_port_open(self):
        try:
            # 检查 Modbus 客户端是否已创建并连接
            if self.client is None:
                print("Modbus 客户端未创建")
                return True
            elif self.client.connect():
                return False
            else:
                print("Modbus 客户端链接失败")
                return True
        except Exception as e:
            print(f"检查串口或 Modbus 客户端时发生错误: {e}")
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

    # 打开串口
    def try_port_open(self, _port, _baudrate=1000000):
        try:
            # if not self.mSerial.isOpen():
            #     self.mSerial.open()
            self.client = ModbusClient(port=_port, baudrate=_baudrate, timeout=0.02, stopbits=1, bytesize=8,
                                  parity='N')  # 看文档，method='rtu'貌似没用
            if self.client == None:
                print(f"Modbus 客户端未创建")
                return False
            client_state = self.client.connect()
            if not client_state:
                print(f"无法连接到 Modbus 服务器: {_port}")
                return False
            print(f"成功连接到 Modbus 服务器: {_port}")
            return True
        except Exception as e:
            print(f"打开串口或连接到 Modbus 服务器时发生错误: {e}")
            return False

    # send data
    # 这个后期可以再改写，因为modbus库通讯速率上不去，所以可能直接再pyserial上改
    def send_data(self, buff):#, isHexSend=False, _baudrate=115200):
        if buff != "":
            num = self.mSerial.write(buff)
            # 非空字符串
            # if isHexSend:
            #     # hex发送
            #     buff = buff.strip()
            #     send_list = []
            #     while buff != '':
            #         try:
            #             num = int(buff[0:2], 16)
            #         except ValueError:
            #             # QMessageBox.critical(self, 'wrong data', '请输入十六进制数据，以空格分开!')
            #             return None
            #         buff = buff[2:].strip()
            #         send_list.append(num)
            #     buff = bytes(send_list)
            # num = self.mSerial.write(buff)
            # self.data_num_sended += num
            # self.lineEdit_2.setText(str(self.data_num_sended))

    def set_register_value(self, id, address, value):
        if self.client is None:
            print("Modbus 客户端未创建")
            return False
        # self.clear_recv_buffer()
        try:
            # 写入单个寄存器的值
            # id表示是哪个驱动器，有驱动器上面的编码决定；address是寄存器地址，即执行什么指令；
            # 指令集如下：0，使能  1，占空比(-100%~100%)  2，同步更新占空比设置  3，泵升电压限制(100.00)= 100*电压值
            # 4，过温保护限制(100.00)= 100*温度值  5，电流(-30.000A~30.000A)=1000*电流值  6，电压(0.00V~200.00V)= 100*电压值
            # 7，H桥温度(-40.00℃~200.80℃)= 100*温度值  8，H桥温度(-40.00℃~200.80℃)= 100*温度值  9，H桥温度(-40.00℃~200.80℃)= 100*温度值
            # 10，H桥温度(-40.80℃~200.00℃)=100*温度值  11，单片机温度(-20.00℃~80.00℃)=100*温度值  12，电流原数据
            # 13，电压原数据  14，H桥温度原数据  15，H桥温度原数据  16，H桥温度原数据  17，H桥温度原数据  18，单片机温度原数据
            # self.client.write_register(slave=id, address=address, value=value, no_response_expected=True)
            # print(time.time())
            response = self.client.write_register(slave=id, address=address, value=value)
            # print(time.time())
            # if response.isError():
            #     print(f"驱动{id}写入寄存器{address}失败")
            #     return False
            # # print(f"成功写入寄存器 {address} 的值为 {value}")
            # print(response.isError())
            return True
        except Exception as e:
            print(f"将{value}写入{id}驱动器的{address}寄存器时发生错误: {e}")
            return False

    def broadcast_set_register(self, address, value):
        # 执行Modbus广播写操作，将值写入指定地址的寄存器。
        # :param address: 寄存器地址
        # :param value: 要写入的值
        # :return: 操作成功返回True，否则返回False
        if self.client is None:
            print("Modbus 客户端未创建")
            return False

        try:
            # 执行广播写操作
            self.client.write_register(slave=0, address=address, value=value, no_response_expected=True)
            # self.client.write_register(slave=0, address=address, value=value)
            # if response.isError():
            #     print(f"广播写入寄存器失败")
            #     return False
            # # print(f"成功广播写入寄存器 {address} 的值为 {value}")
            return True
        except Exception as e:
            print(f"执行广播写操作时发生错误: {e}")
            return False

    # def read_register_value(self, id, address, count):
    #     # 从指定的Modbus从设备读取寄存器值。
    #     # :param id: 从设备ID
    #     # :param address: 寄存器起始地址
    #     # :param count: 要读取的寄存器数量
    #     # :return: 成功读取时返回寄存器值，否则返回None
    #     if self.client is None:
    #         print("Modbus 客户端未创建")
    #         return None
    #
    #     try:
    #         # 从Modbus从设备读取寄存器值
    #         result = self.client.read_holding_registers(slave=id, address=address, count=count)
    #         if result.isError():
    #             # 处理错误
    #             print(f"读取驱动{id}错误")
    #             return None
    #         # 确保有足够的寄存器值
    #         if len(result.registers) < 1:
    #             print("寄存器值不足，无法获取数据。")
    #             return None
    #         # 只需要读取第一个寄存器的值
    #         register_value = result.registers[0]
    #         return register_value
    #     except Exception as e:
    #         print(f"读取寄存器时发生异常: {e}")
    #         return None

    def read_current_value(self, id, count):
        try:
            result = self.client.read_holding_registers(address=5, count=count, slave=id)
            if result.isError():
                # 处理错误
                print(f"读取寄存器{id}的电流错误: {result}")
                return None
            # 确保有足够的寄存器值
            if len(result.registers) < 1:
                print(f"寄存器{id}的电流值不足，无法获取数据")
                return None
            register_0 = result.registers[0]
            # 将 uint16 转换为 int16
            if register_0 & 0x8000:  # 如果最高位为 1，表示是负数
                register_0 = -(0x10000 - register_0)
            current = register_0 / 1000
            return current
        except Exception as e:
            print(f"读取电流发生异常: {e}")
            return None

    # 这个不常用，测的是母线电压
    def read_voltage_value(client, id, count):
        try:
            result = client.read_holding_registers(address=6, count=count, slave=id)
            if result.isError():
                # 处理错误
                print(f"读取错误: {result}")
                return None
            # 确保有足够的寄存器值
            if len(result.registers) < 1:
                print("寄存器值不足，无法获取数据。")
                return None
            register_0 = result.registers[0]
            print("voltage_register_0", register_0)
            # 将 uint16 转换为 int16
            if register_0 & 0x8000:  # 如果最高位为 1，表示是负数
                register_0 = -(0x10000 - register_0)
            current = register_0
            return current
        except Exception as e:
            print(f"发生异常: {e}")
            return None