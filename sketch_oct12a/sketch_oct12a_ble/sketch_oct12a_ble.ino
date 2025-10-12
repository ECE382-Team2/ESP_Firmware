#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

// BLE Service and Characteristic UUIDs
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcdef1"

// Set up the BLE server
BLECharacteristic *pCharacteristic;
BLEServer *pServer;

bool deviceConnected = false;

// UART1 pins for the ESP32-S3
#define RXD1 18 // A0
#define TXD1 17 // A1 but transmit to PSoC not implemented yet


// This function is called when a client connects/disconnects
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
    }

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
    }
};

void setup() {
  Serial.begin(9600);

  // Initialize BLE
  BLEDevice::init("ESP32_COINTF_WIRELESS_OREO"); // Name of the device

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create a service
    BLEService *pService = pServer->createService(SERVICE_UUID);

    // Create a characteristic
    pCharacteristic = pService->createCharacteristic(
                          CHARACTERISTIC_UUID,
                          BLECharacteristic::PROPERTY_READ |
                          BLECharacteristic::PROPERTY_WRITE |
                          BLECharacteristic::PROPERTY_NOTIFY
                      );
    pCharacteristic->setValue("Hello from ESP32");

    // Start the service
    pService->start();

    // Start advertising the service
    BLEAdvertising *pAdvertising = pServer->getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->start();

    Serial.println("Waiting for a client to connect...");

  while (!Serial);

  Serial.println("ESP32 UART Receiver Initialized");

  Serial1.begin(9600, SERIAL_8N1, RXD1, TXD1); // 8 data, no parity, 1 stop bit
}

void loop() {
  // Read from Serial1 (external device)
  if (deviceConnected) {    
    String in = Serial1.readStringUntil('\n');
    Serial.print("Received via UART1: ");
    Serial.println(in); 

    // Send data every second
    pCharacteristic->setValue(in);
    pCharacteristic->notify();

  }
}