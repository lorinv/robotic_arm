import sys
import serial

AX_WRITE_DATA = 3
AX_READ_DATA = 4

motors_mins = [0,512,512,512,480]
motors_maxs = [1023, 870, 980, 870, 870]

from time import sleep

class ArmController(object):

    """docstring foArm_Controllerme"""
    def __init__(self, port="/dev/ttyUSB0"):
        self.motors = [512,512,512,512,512]
        self.s = serial.Serial()               # create a serial port object
        self.s.baudrate = 38400
        try:              # baud rate, in bits/second
            self.s.port = port           # this is whatever port your are using
            self.s.open()
        except:
            self.s.port = "/dev/ttyUSB0"   
            self.s.open()        # this is whatever port your are using

        self.initialize()

    def initialize(self):
        self.s.flushInput()
        for i in range(30):
            temp_position = 512
            offset = temp_position / 256
            location = temp_position % 256
            self.setReg(1, 30, ((location%265),offset))                    
            self.setReg(2, 30, ((location%265),offset))                    
            self.setReg(3, 30, ((location%265),offset))                    
            self.setReg(4, 30, ((location%265),offset))                    
            self.setReg(5, 30, ((location%265),offset))                    
            sleep(1)       

    def set_joint(self, motor_id, position):    
        self.s.flushInput()
        if position > motors_maxs[motor_id - 1]:
            position = motors_maxs[motor_id - 1]

        elif position < motors_mins[motor_id - 1]:
            position = motors_mins[motor_id - 1]

        for i in range(abs(self.motors[motor_id - 1] - position)):
            if self.motors[motor_id - 1] - position < 0:
                temp_position = self.motors[motor_id - 1] + i
            else:
                temp_position = self.motors[motor_id - 1] - i

            offset = temp_position / 256
            location = temp_position % 256
            self.setReg(motor_id, 30, ((location%265),offset))                    
            sleep(.01)        

        self.motors[motor_id - 1] = position
        return position


    def set_all_joints(self, positions):
        for i, pos in enumerate(position):
            self.set_joint(i + 1, pos)


    def get_joint_pose(self, id):   
        pass

    def get_all_posees(self):
        pass

    def setReg(self, ID,reg,values):
        self.s.flushInput()
        length = 3 + len(values)
        checksum = 255-((ID+length+AX_WRITE_DATA+reg+sum(values))%256)          
        self.s.write(chr(0xFF)+chr(0xFF)+chr(ID)+chr(length)+chr(AX_WRITE_DATA)+chr(reg))
        for val in values:
           self.s.write(chr(val))
        self.s.write(chr(checksum))

    def getReg(self, index, regstart, rlength):
       self.s.flushInput()   
       checksum = 255 - ((6 + index + regstart + rlength)%256)
       self.s.write(chr(0xFF)+chr(0xFF)+chr(index)+chr(0x04)+chr(AX_READ_DATA)+chr(regstart)+chr(rlength)+chr(checksum))
       vals = list()
       self.s.read()   # 0xff
       self.s.read()   # 0xff
       self.s.read()   # ID
       length = ord(self.s.read()) - 1
       self.s.read()   # toss error    
       while length > 0:
           vals.append(ord(self.s.read()))
           length = length - 1
       if rlength == 1:
           return vals[0]
       return vals

if __name__ == "__main__":  
    arm = ArmController()    

    while 1:
        motor_id = int(raw_input("Motor ID: "))
        position = int(raw_input("Position: "))        
        arm.set_joint(motor_id, position)        
        