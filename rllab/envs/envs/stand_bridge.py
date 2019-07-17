import argparse
import copy
import shlex
import subprocess
from time import time, ctime

from pyfirmata import Arduino, ArduinoMega, util

class StandBridge(object):
    def __init__(self, port, timeout=0):
        self._port = port
        self._state = {}

        start_setup = time()

        self._board = ArduinoMega(port)

        self.pin_a0 = 0  # Rod pressure
        self.pin_a1 = 1  # Base Pressure
        self.pin_a2 = 2  # System Pressure
        self.pin_a3 = 3  # Load Sense Pressure
        self.pin_a4 = 4  # Reservoir Temperature
        self.pin_a5 = 5  # Cylinder Position

        self._pwmPin2 = self._board.get_pin("d:2:p")  # Base coil down
        self._pwmPin3 = self._board.get_pin("d:3:p")  # Rod coil up

        self._iterator = util.Iterator(self._board)
        self._iterator.start()

        self._board.analog[self.pin_a0].enable_reporting()  # Need some-time after this
        self._board.analog[self.pin_a1].enable_reporting()  # Need some-time after this
        self._board.analog[self.pin_a2].enable_reporting()  # Need some-time after this
        self._board.analog[self.pin_a3].enable_reporting()  # Need some-time after this
        self._board.analog[self.pin_a4].enable_reporting()  # Need some-time after this
        self._board.analog[self.pin_a5].enable_reporting()  # Need some-time after this

        print("This run starts at: ", ctime())

        while self._board.analog[self.pin_a0].read() is None:
            pass

        while self._board.analog[self.pin_a1].read() is None:
            pass

        while self._board.analog[self.pin_a2].read() is None:
            pass

        while self._board.analog[self.pin_a3].read() is None:
            pass

        while self._board.analog[self.pin_a4].read() is None:
            pass

        while self._board.analog[self.pin_a5].read() is None:
            pass

        print("Time taken to setup Arduino: {:.4f} s".format(time() - start_setup))

    def read_state(self):
        self._state["Rod_Pressure"] = self._board.analog[self.pin_a0].read()
        self._state["Base_Pressure"] = self._board.analog[self.pin_a1].read()
        self._state["System_Pressure"] = self._board.analog[self.pin_a2].read()
        self._state["Load_Sense_Pressure"] = self._board.analog[self.pin_a3].read()
        self._state["Reservoir_Temperature"] = self._board.analog[self.pin_a4].read()
        self._state["Height"] = self._board.analog[self.pin_a5].read()
        return self._state

    def send_commands(self, duty_cycle_down, duty_cycle_up):
        assert (
            duty_cycle_down == 0.0 or duty_cycle_up == 0.0
        ), "down_cycle and up_cycle commanded at same time"
        assert (
            duty_cycle_down >= 0.0 and duty_cycle_up >= 0.0
        ), "negative duty cycle commanded"

        duty_cycle_down = 1.0 if (duty_cycle_down > 1.0) else duty_cycle_down
        duty_cycle_up = 1.0 if (duty_cycle_up > 1.0) else duty_cycle_up

        self._pwmPin2.write(duty_cycle_down)
        self._pwmPin3.write(duty_cycle_up)

    def send_default_commands(self):
        self.send_commands(duty_cycle_down=0.0, duty_cycle_up=0.0)

    def close(self):
        self.send_default_commands()
        self._board.analog[self.pin_a0].disable_reporting()
        self._board.analog[self.pin_a1].disable_reporting()
        self._board.analog[self.pin_a2].disable_reporting()
        self._board.analog[self.pin_a3].disable_reporting()
        self._board.analog[self.pin_a4].disable_reporting()
        self._board.analog[self.pin_a5].disable_reporting()
        self._iterator = None
        self._board.exit()

        print("Arduino Board Closed successfully!")


def pause_operation(bridge):
    bridge.send_default_commands()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", default=6, help="Arduino timeout", type=int)

    args = parser.parse_args()

    find_usbid_cmd = shlex.split("ls /dev/serial/by-id/")
    usb_id = "/dev/serial/by-id/" + subprocess.check_output(find_usbid_cmd).decode()
    usb_id = usb_id[0:-1]

    bridge = StandBridge(port=usb_id, timeout=args.timeout)

    # sun hydraulics XMD and solenoids
    # defining duty cycle to be command
    duty_cycle_down = 0.7
    duty_cycle_up = 0.0
    bridge.send_commands(duty_cycle_down, duty_cycle_up)

    print("=== STATE ===")
    for i in range(int(5e5)):
        try:
            state = copy.deepcopy(bridge.read_state())
            print(state)

        except KeyboardInterrupt:
            print("exit bridge")
            bridge.close()
            break

    bridge.close()
    print("=== end ===")


if __name__ == "__main__":
    main()
