#------------------------------------------------------------------------
#RFCOMM_BT_Client
#------------------------------------------------------------------------
#Allows text to be published by the client and immediately recieved on the
#server through the RFCOMM sockets - uses a pre-paired bluetooth connection
#between two raspberry pis
#Prior to use:
#1) Modify /etc/systemd/system/bluetooth.target.wants/bluetooth.service file
#to add -C after bluetoothhd
#2) Add serial port serice sudo sdptool add SP
#3) Run sudo hciconfig hci0 piscan
#4) Run this file from root with sudo and python3
#------------------------------------------------------------------------
#Author   :   Luke Holland
#Date   :   21st December 2020
#Modified from the bluetooth file examples by Albert Huang
#------------------------------------------------------------------------


from bluetooth import *
import sys
import time
import math

if sys.version < '3':
    input = raw_input

addr = "B8:27:EB:B0:03:7F"

if len(sys.argv) < 2:
    print("no device specified.  Searching all nearby bluetooth devices for")
    print("the SampleServer service")
else:
    addr = sys.argv[1]
    print("Searching for SampleServer on %s" % addr)

# search for the SampleServer service
uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
service_matches = find_service( uuid = uuid, address = addr )

if len(service_matches) == 0:
    print("couldn't find the SampleServer service =(")
    sys.exit(0)

first_match = service_matches[0]
port = first_match["port"]
name = first_match["name"]
host = first_match["host"]
print("port : " + str(port))
print("name : " + str(name))
print("host : " + str(host))

print("connecting to \"%s\" on %s" % (name, host))

# Create the client socket
sock=BluetoothSocket( RFCOMM )
print(port)
sock.connect((host, 1))

print("connected.  type stuff")
i=0

while True:
    #waits for a time interval and then access all relevant data
    #read from the ChairState.txt database 
    PresenceFile = open("ChairState.txt", "r")
    UserStateFile = open("UserState.txt","r")

    data = str(PresenceFile.read()) + "," + str(UserStateFile.read())
    print(data)
    sock.send(data)
    time.sleep(15)

sock.close()
