// UART1 pins for the ESP32-S3
#define RXD1 18 // A0
#define TXD1 17 // A1 but transmit to PSoC not implemented yet

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("ESP32 UART Receiver Initialized");

  Serial1.begin(9600, SERIAL_8N1, RXD1, TXD1); // 8 data, no parity, 1 stop bit
}

void loop() {
  // Read from Serial1 (external device)
  // We may want to switch to sending numbers instead of strings later, though
  if (Serial1.available()) {
    String in = Serial1.readStringUntil('\n');
    //Serial.print("Received via UART1: ");
    Serial.println(in);

    // TODO (formatting)
    // - Send 1 vs 255 for mode
    // - Remove spaces
  }
}