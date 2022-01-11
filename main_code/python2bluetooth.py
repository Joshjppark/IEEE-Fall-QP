
import bluepy.btle as btle
import time


def main():
    try:
        while True:
            text = input("Enter input: ")
            send2bluetooth(address, text)
            time.sleep(.1)
    except KeyboardInterrupt:
        print("\nProgram closing...")


def send2bluetooth(address, text):
    try:
        hm10 = btle.Peripheral(address)
        service = hm10.getServiceByUUID("0000ffe0-0000-1000-8000-00805f9b34fb") # service
        characteristic = service.getCharacteristics()[0]
        characteristic.write(bytes(text, "utf-8"))
        hm10.disconnect()
    except:
        print('Failed to connect to HM10; continuing...')
        return

if __name__ == '__main__':
    address = "30:E2:83:8D:78:20"


    main()