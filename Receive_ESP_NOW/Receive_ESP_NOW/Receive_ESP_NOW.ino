/*
  Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/esp-now-esp32-arduino-ide/  
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <esp_now.h>
#include <WiFi.h>

// Char to receive data
char inChar;

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  memcpy(&inChar, incomingData, sizeof(inChar));

  // Build payload string from incoming bytes
  String payload;
  for (int i = 0; i < len; ++i) payload += (char)incomingData[i];
  payload.trim();

  // Expect format: "flag, C1, C2, C3, C4"
  int commaPos = payload.indexOf(',');
  if (commaPos < 0) return;
  int flag = payload.substring(0, commaPos).toInt();
  if (flag == 1) return; // do not print anything

  // Parse the four comma separated numbers
  float vals[4] = {0,0,0,0};
  String rest = payload.substring(commaPos + 1);
  rest.trim();
  for (int i = 0; i < 4; ++i) {
    int nextComma = rest.indexOf(',');
    String token;
    if (nextComma >= 0) {
      token = rest.substring(0, nextComma);
      rest = rest.substring(nextComma + 1);
    } else {
      token = rest;
      rest = "";
    }
    token.trim();
    vals[i] = token.toFloat();
  }

  // Linear model coefficients (adjust constants as needed)
  const float Fx_coef[4] = {0.001153781f, -0.000735835f, -0.000403246f, 0.054144917f};
  const float Fy_coef[4] = {-0.000745378f, -0.001211443f,  0.001450559f, -0.005559095f};
  const float Fz_coef[4] = {0.000268684f, 0.001540816f, 0.00053931f,  0.105507336f};

  float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
  for (int i = 0; i < 4; ++i) {
    Fx += Fx_coef[i] * vals[i] + -766.3605007f;
    Fy += Fy_coef[i] * vals[i] + 86.29901195;
    Fz += Fz_coef[i] * vals[i] + -1528.151147;
  }

  Serial.print("Fx: "); Serial.print(Fx, 4);
  Serial.print(", Fy: "); Serial.print(Fy, 4);
  Serial.print(", Fz: "); Serial.println(Fz, 4);
}
 
void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);
  
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
}
 
void loop() {

}