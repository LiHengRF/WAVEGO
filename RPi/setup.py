#!/usr/bin/python3
# File name   : setup.py for WAVEGO
# Date        : 2022/1/5
# Modified    : 2025 - Fixed for Debian 12+ (PEP 668 compliance)

import os
import time
import re

curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)

# Check if running as root
if os.geteuid() != 0:
    print("Error: This script must be run with sudo!")
    print("Usage: sudo python3 setup.py")
    exit(1)

def replace_num(file, initial, new_num):  
    newline = ""
    str_num = str(new_num)
    with open(file, "r") as f:
        for line in f.readlines():
            if(line.find(initial) == 0):
                line = (str_num + '\n')
            newline += line
    with open(file, "w") as f:
        f.writelines(newline)

# System update
for x in range(1, 4):
    if os.system("apt update") == 0:
        break

for x in range(1, 4):
    if os.system("apt -y dist-upgrade") == 0:
        break

for x in range(1, 4):
    if os.system("apt clean") == 0:
        break

# Upgrade pip (with --break-system-packages flag for PEP 668)
for x in range(1, 4):
    if os.system("pip3 install --break-system-packages -U pip") == 0:
        break

# Install system dependencies (changed python-dev to python3-dev)
for x in range(1, 4):
    if os.system("apt-get install -y python3-dev python3-pip libfreetype6-dev libjpeg-dev build-essential") == 0:
        break

for x in range(1, 4):
    if os.system("apt-get install -y i2c-tools") == 0:
        break

for x in range(1, 4):
    if os.system("apt-get install -y python3-smbus") == 0:
        break

# Install Python packages (with --break-system-packages flag)
for x in range(1, 4):
    if os.system("pip3 install --break-system-packages pyserial") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple pyserial") == 0:
        break

for x in range(1, 4):
    if os.system("pip3 install --break-system-packages flask") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple flask") == 0:
        break

for x in range(1, 4):
    if os.system("pip3 install --break-system-packages flask_cors") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple flask_cors") == 0:
        break

for x in range(1, 4):
    if os.system("pip3 install --break-system-packages websockets") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple websockets") == 0:
        break

# Modify boot config
try:
    replace_num("/boot/config.txt", '[all]', '[all]\nenable_uart=1\ngpu_mem=128')
except:
    print('Failed to modify /boot/config.txt [all] section, try again')

try:
    replace_num("/boot/config.txt", 'camera_auto_detect=1', '#camera_auto_detect=1\nstart_x=1')
except:
    print('Failed to modify camera_auto_detect, try again')

try:
    replace_num("/boot/config.txt", 'camera_auto_detect=1', '#camera_auto_detect=1')
except:
    print('Failed to comment camera_auto_detect, try again')

# Modify cmdline.txt
try:
    CMDLINE_FILE = open('/boot/cmdline.txt', 'r')
    OLD_LINES = CMDLINE_FILE.readlines()
    CMDLINE_FILE.close()

    CMDLINE_FILE = open('/boot/cmdline.txt', 'w+')
    for EACH_LINE in OLD_LINES:
        NEW_LINES = re.sub('console=serial0,115200', '', EACH_LINE)
        CMDLINE_FILE.writelines(NEW_LINES)
    CMDLINE_FILE.close()
except Exception as e:
    print(f'Failed to modify /boot/cmdline.txt: {e}')

# Install OpenCV (using newer compatible version)
for x in range(1, 4):
    if os.system("pip3 install --break-system-packages opencv-contrib-python") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple opencv-contrib-python") == 0:
        break

# Install numpy (compatible version)
for x in range(1, 4):
    if os.system("pip3 uninstall -y numpy") == 0:
        break

for x in range(1, 4):
    if os.system("pip3 install --break-system-packages numpy") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple numpy") == 0:
        break

# Install HDF5 and other libraries
for x in range(1, 4):
    if os.system("apt-get -y install libhdf5-dev libhdf5-serial-dev libatlas-base-dev") == 0:
        break

# Install additional Python packages
for x in range(1, 4):
    if os.system("pip3 install --break-system-packages imutils zmq pybase64 psutil") == 0:
        break
    elif os.system("pip3 install --break-system-packages -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple imutils zmq pybase64 psutil") == 0:
        break

# Install network tools
for x in range(1, 4):
    if os.system("apt-get install -y util-linux procps hostapd iproute2 iw haveged dnsmasq") == 0:
        break

# Clone and install create_ap
for x in range(1, 4):
    if os.system("cd " + thisPath + " && cd .. && git clone https://github.com/oblique/create_ap") == 0:
        break

try:
    os.system("cd " + thisPath + " && cd .. && cd create_ap && make install")
except:
    pass

# Add webServer to startup
try:
    replace_num('/etc/rc.local', 'exit 0', 'cd ' + thisPath + ' && python3 webServer.py &\nexit 0')
except Exception as e:
    print(f'Failed to modify /etc/rc.local: {e}')

print('Setup completed!')

os.system("reboot")
