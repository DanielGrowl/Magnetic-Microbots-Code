#   Copyright (c) 2019 Universea Author. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math

class PowerSupply():

    def __init__(self):

        self.hx_address = 0x01
        self.hy_address = 0x03
        self.hz_address = 0x02


    def calc_crc(self,string):
        data = string#bytearray.fromhex(string)
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if ((crc & 1) != 0):
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        #return hex(((crc & 0xff) << 8) + (crc >> 8))
        return ((crc & 0xff)),(crc >> 8)

    def set_output(self,address,current):

        current = int(current*1000)
        # if address == self.hx_address:
        #     current = int(1E-06*current**2 + 1.0149*current + 29.678)
        # if address == self.hy_address:
        #     current = int(3E-06*current**2 + 1.0445*current + 35.427)
        # if address == self.hz_address:
        #     current = int(5E-06*current**2 + 1.04459*current + 17.045)

        current = (current).to_bytes(4, byteorder='big',signed=True)

        send_list = []
        send_list.append(address)
        send_list.append(0x10)
        send_list.append(0x20)
        send_list.append(0x30)
        send_list.append(0x00)
        send_list.append(0x02)
        send_list.append(0x04)
        send_list.append(current[0])
        send_list.append(current[1])
        send_list.append(current[2])
        send_list.append(current[3])
        crc_h,crc_l = self.calc_crc(bytes(send_list))
        send_list.append(crc_h)
        send_list.append(crc_l)
        return send_list

    def set_current_mode(self,address):

        send_list = []
        send_list.append(address)
        send_list.append(0x06)
        send_list.append(0x60)
        send_list.append(0x60)
        send_list.append(0xff)
        send_list.append(0xfd)
        crc_h,crc_l = self.calc_crc(bytes(send_list))
        send_list.append(crc_h)
        send_list.append(crc_l)
        return send_list

    def open_output(self,address):
        send_list = []
        send_list.append(address)
        send_list.append(0x06)
        send_list.append(0x60)
        send_list.append(0x40)
        send_list.append(0x00)
        send_list.append(0xff)
        crc_h,crc_l = self.calc_crc(bytes(send_list))
        send_list.append(crc_h)
        send_list.append(crc_l)
        return send_list

    def close_output(self,address):

        send_list = []
        send_list.append(address)
        send_list.append(0x06)
        send_list.append(0x60)
        send_list.append(0x40)
        send_list.append(0x00)
        send_list.append(0x00)
        crc_h,crc_l = self.calc_crc(bytes(send_list))
        send_list.append(crc_h)
        send_list.append(crc_l)
        return send_list


if __name__ == "__main__":

    power = PowerSupply()
    # send = power.set_current_mode(0x06)
    # hex_numbers = [hex(num) for num in send]
    # print(hex_numbers)
    send = power.open_output(0x04)
    hex_numbers = [hex(num) for num in send]
    print(hex_numbers)
    # send = power.set_output(0x06,2)
    # hex_numbers = [hex(num) for num in send]
    # print(hex_numbers)
    # print(send)