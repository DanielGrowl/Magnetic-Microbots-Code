import logging
import time
import math
import UartSerial
import numpy as np


class Drive:
    def __init__(self, _uart_port, _power_supply):
        self.power = _power_supply
        self.uart_port = _uart_port
        self.power_state = False

    # 基础性自定义函数部分
    # 占空比到有符号init16的转化
    def calculate_int16_from_duty_cycle(self, duty_cycle):
        # 将PWM占空比转换为int16值。
        # :param duty_cycle: PWM占空比，范围在-1.0到1.0
        # :return: 对应的int16值
        # 将占空比乘以32768并转换为整数
        int16_value = int(duty_cycle * 32768)
        # 转换为16位有符号整数
        int16_value = np.int16(int16_value)
        return int16_value

    # 基础性自定义函数，包括了占空比到有符号init16的转化
    def calculate_pwm_and_uint16(self, target_current, supply_voltage, resistance):
        # 计算PWM占空比和对应的uint16值以达到目标电压。
        # :param target_voltage: 目标电压（伏特）
        # :param supply_voltage: 供电电压（伏特）
        # :param resistance: 线圈电阻（欧姆）
        # :return: PWM占空比和对应的uint16值

        # 计算最大电流
        max_current = supply_voltage / resistance
        if abs(target_current) >= max_current:
            target_current = max_current
            print("超过电流上限")
        # 计算PWM占空比
        duty_cycle = target_current / max_current
        # 占空比先转化为init16型，这也是驱动能够读的
        pwm_value_int16 = int(duty_cycle * 32767)
        # 再转为uint16型，这是因为发送只能发送0-65535
        if pwm_value_int16 >= 0:
            pwm_value_uint16 = pwm_value_int16
        else:
            pwm_value_uint16 = pwm_value_int16 & 0xffff
        return duty_cycle, pwm_value_uint16

    # 初始化，包括了驱动器使能和归零
    def init_power(self):
        # 先set设置电流，再open执行
        # set current
        self.uart_port.broadcast_set_register(0, 1)
        # print(time.time())
        self.set_currents(0, 0, 0, 0, 0, 0)
        # print(time.time())
        print("初始化成功")
        # set power inti state
        # 代表了电源模块初始化已完成
        self.power_state = True

    # 关闭，包括了归零和驱动器关闭
    def uninit_power(self):
        # close current
        try:
            self.set_currents(0, 0, 0, 0, 0, 0)
            self.uart_port.broadcast_set_register(0, 0)
            # set power inti state
            # 代表了电源模块已关闭
            self.power_state = False
            return True
        except Exception as e:
            print(f"关闭port发生错误: {e}")
            return False

    # 这边的用法之后得改，还得想想怎么改，怎么循环，可能得改成roll和pitch的形式，一个决定滚的速度一个决定朝向的改变
    def motion_up(self):
        pass
        # self.set_current(self.power.MX1_address, 5, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MY1_address, 5, -50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ2_address, 0, 50)

    def motion_down(self):
        pass
        # self.set_current(self.power.MX1_address, 0, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MX2_address, 10, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MY1_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MY2_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ1_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ2_address, 0, 50)

    def motion_left(self):
        pass
        # self.set_current(self.power.MX1_address, 0, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MX2_address, 0, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MY1_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MY2_address, 10, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ1_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ2_address, 0, 50)

    def motion_right(self):
        pass
        # self.set_current(self.power.MX1_address, 0, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MX2_address, 0, 100)
        # time.sleep(0.003)
        # self.set_current(self.power.MY1_address, 10, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MY2_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ1_address, 0, 50)
        # time.sleep(0.003)
        # self.set_current(self.power.MZ2_address, 0, 50)

    # 停止运动，包含了驱动器清零
    def motion_stop(self):
        self.set_currents(0,0,0,0,0,0)

    # 这个用来设置单个驱动器PWM
    # 变量：目标电流、供给电压、系统电阻(包括线圈和驱动)、驱动编号、寄存器地址位
    def set_current_single(self, target_current, supply_voltage, resistance, id, address):
        desired_dutycycle, setvalue = self.calculate_pwm_and_uint16(target_current=target_current, supply_voltage=supply_voltage,
                                                 resistance=resistance)
        return self.uart_port.set_register_value(id, address, setvalue)

    # 这个用来同步更新驱动器PWM
    def enable_current(self):
        return self.uart_port.broadcast_set_register(address=2, value=1)

    # 这个用来读取单个驱动的电流
    # 变量：驱动编号、读取的位数，一般不用管
    def read_current_single(self, id, count):
        return self.uart_port.read_current_value(id, count)

    # 这个用来同时设置所有的电流并同步更新
    # 变量：X1,X2,Y1,Y2,Z1,Z2电流
    def set_currents(self, IX1, IX2, IY1, IY2, IZ1, IZ2):
        # print(f"电流设置：IX1({IX1:.3f}),IX2({IX2:.3f}),IY1({IY1:.3f}),IY2({IY2:.3f}),IZ1({IZ1:.3f}),IZ2({IZ2:.3f})")
        # IX1
        supply_voltage_x = 30
        resistance_x1 = 2.7
        resistance_x2 = 2.67
        self.set_current_single(IX1, supply_voltage_x, resistance_x1, 6, 1)
        time.sleep(0.001)
        # IX2
        self.set_current_single(IX2, supply_voltage_x, resistance_x2, 5, 1)
        time.sleep(0.001)
        # IY1
        supply_voltage_y = 15
        resistance_y1 = 1.31
        resistance_y2 = 1.275
        self.set_current_single(IY1, supply_voltage_y, resistance_y1, 4, 1)
        time.sleep(0.001)
        # IY2
        self.set_current_single(IY2, supply_voltage_y, resistance_y2, 3, 1)
        time.sleep(0.001)
        # IZ1
        supply_voltage_z = 5
        resistance_z1 = 0.571
        resistance_z2 = 0.562
        self.set_current_single(IZ1, supply_voltage_z, resistance_z1, 2, 1)
        time.sleep(0.001)
        # IZ2
        self.set_current_single(IZ2, supply_voltage_z, resistance_z2, 1, 1)
        time.sleep(0.001)
        self.enable_current()
        # time.sleep(0.001)
        # IX1_read, IX2_read, IY1_read, IY2_read, IZ1_read, IZ2_read = self.read_currents()
        # print(f"电流读取：IX1({IX1_read}),IX2({IX2_read}),IY1({IY1_read}),IY2({IY2_read}),IZ1({IZ1_read}),IZ2({IZ2_read})")

    def read_currents(self):
        IX1 = self.read_current_single(6, 2)
        time.sleep(0.001)
        IX2 = self.read_current_single(5, 2)
        time.sleep(0.001)
        IY1 = self.read_current_single(4, 2)
        time.sleep(0.001)
        IY2 = self.read_current_single(3, 2)
        time.sleep(0.001)
        IZ1 = self.read_current_single(2, 2)
        time.sleep(0.001)
        IZ2 = self.read_current_single(1, 2)
        time.sleep(0.001)
        return IX1, IX2, IY1, IY2, IZ1, IZ2

    # def open_current(self, address):
    #     # if self.uart_port.is_port_open():#self.uart_port.mSerial.port, self.uart_port.mSerial.baudrate):
    #     send_data = self.power.openoutput(address)
    #     self.uart_port.send_data(send_data)
    #
    # def close_current(self, address):
    #     # if self.uart_port.is_port_open(self.uart_port.mSerial.port, self.uart_port.mSerial.baudrate):
    #     send_data = self.power.closeoutput(address)
    #     self.uart_port.send_data(send_data)
