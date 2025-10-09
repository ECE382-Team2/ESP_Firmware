// Adafruit QT Py ESP32-S3 External GPIO Toggle
//
// This sketch toggles an external General Purpose Input/Output (GPIO) pin
// every second.
//
// We are using GPIO 3, which is the pin labeled 'A3' on the QT Py S3 pinout.
// You can connect an external component (like an LED+resistor) here.

// Define the GPIO pin we will use.
// Change this number to any other available GPIO pin if needed.
const int EXTERNAL_GPIO_PIN = 9;

void setup() {
  // Initialize the selected GPIO pin as an output.
  pinMode(EXTERNAL_GPIO_PIN, OUTPUT);

  // Initialize Serial Communication for debugging (optional but recommended)
  Serial.begin(115200);
  Serial.println("External GPIO Toggle Running...");
}

void loop() {
  // 1. Turn the GPIO pin HIGH (usually 3.3V)
  digitalWrite(EXTERNAL_GPIO_PIN, HIGH);
  Serial.println("GPIO 3: HIGH");
  // Wait for 1000 milliseconds (1 second)
  delay(1000);

  // 2. Turn the GPIO pin LOW (usually 0V/GND)
  digitalWrite(EXTERNAL_GPIO_PIN, LOW);
  Serial.println("GPIO 3: LOW");
  // Wait for 1000 milliseconds (1 second)
  delay(1000);
}
