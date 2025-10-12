#include "BluetoothSerial.h"

class BTReceiver {
private:
    BluetoothSerial SerialBT;
    String deviceName;
    
public:
    BTReceiver(String name = "ESP32_BT") {
        deviceName = name;
    }
    
    void begin() {
        if (!SerialBT.begin(deviceName)) {
            Serial.println("Bluetooth initialization failed!");
            return;
        }
        Serial.begin(115200);
        Serial.println("Bluetooth Started! Waiting for connections...");
    }
    
    void receiveAndPrint() {
        if (SerialBT.available()) {
            String received = SerialBT.readStringUntil('\n');
            Serial.print("Received: ");
            Serial.println(received);
        }
    }
    
    bool isConnected() {
        return SerialBT.hasClient();
    }
};

// Usage example
BTReceiver btReceiver("MyESP32");

void setup() {
    btReceiver.begin();
}

void loop() {
    btReceiver.receiveAndPrint();
    delay(10);
}