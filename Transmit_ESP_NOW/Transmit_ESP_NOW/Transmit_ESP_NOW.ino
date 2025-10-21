/*
  Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/esp-now-esp32-arduino-ide/
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/
#include <esp_now.h>
#include <WiFi.h>

// UART1 pins for the ESP32-S3
#define RXD1 41 // A0
#define TXD1 40 // A1 but transmit to PSoC not implemented yet

HardwareSerial mySerial(2);

// REPLACE WITH YOUR RECEIVER MAC Address
uint8_t broadcastAddress[] = {0xb4, 0x3a, 0x45, 0xb0, 0xca, 0x5c}; // ESP with duct tape  receives

// Char for transmitting data
char outChar;

esp_now_peer_info_t peerInfo;

// callback when data is sent
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  // Serial.print("\r\nLast Packet Send Status:\t");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}
 
void setup() {
  // Init Serial Monitor
  Serial.begin(9600);
 
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  (esp_now_register_send_cb(OnDataSent));
  
  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }

  
  mySerial.begin(9600, SERIAL_8N1, RXD1, TXD1); // 8 data, no parity, 1 stop bit
}
 
void loop() {  

  if (mySerial.available()) {
    outChar = mySerial.read();
    //Serial.print("Received via UART1: ");

    // Send message via ESP-NOW
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *) &outChar, sizeof(outChar));
    
    if (result == ESP_OK) {
      Serial.println("Sent with success");
    }
    else {
      //Serial.println("Error sending the data");
      Serial.print(outChar);

    }
    
  } 

}