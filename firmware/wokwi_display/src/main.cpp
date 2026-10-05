#include <Arduino.h>
#include <Adafruit_ILI9341.h>
#include <math.h>
#include <string.h>

namespace {
constexpr size_t kLineCapacity = 192;
char lineBuffer[kLineCapacity];
size_t lineLength = 0;
bool lineOverflow = false;
Adafruit_ILI9341 display(5, 27);

void label(int x, int y, const char *text, uint16_t color, int size = 2) {
  display.setCursor(x, y);
  display.setTextColor(color);
  display.setTextSize(size);
  display.print(text);
}

void drawReading(char **fields, float prediction, float reference, bool extended) {
  display.fillScreen(ILI9341_BLACK);
  label(10, 10, "PPG + ACC / HR", ILI9341_CYAN);
  label(10, 36, "AI: HOST CPU | ESP32: DISPLAY", ILI9341_LIGHTGREY, 1);
  display.setCursor(10, 60);
  display.setTextColor(ILI9341_CYAN);
  display.setTextSize(5);
  display.print(prediction, 2);
  label(185, 82, "BPM", ILI9341_CYAN);
  char text[80];
  snprintf(text, sizeof(text), "ECG ref: %.2f BPM", reference);
  label(10, 116, text, ILI9341_GREEN);
  snprintf(text, sizeof(text), "Abs err: %.2f BPM", fabsf(prediction - reference));
  label(10, 143, text, ILI9341_YELLOW);
  snprintf(text, sizeof(text), "%s | %s seed %s", fields[2], extended ? fields[5] : "legacy", extended ? fields[6] : "42");
  label(10, 173, text, ILI9341_WHITE, 1);
  label(10, 195, fields[0], ILI9341_LIGHTGREY, 1);
  snprintf(text, sizeof(text), "%s / offline recorded window", fields[1]);
  label(10, 219, text, ILI9341_ORANGE, 1);
}

void printError(const char *message) {
  Serial.print("ERR|");
  Serial.println(message);
}

bool parseBpm(const char *text, float &value) {
  char *end = nullptr;
  value = strtof(text, &end);
  return end != text && *end == '\0' && isfinite(value);
}

void showReading(char *line) {
  char *fields[7] = {};
  size_t count = 0;
  char *field = strtok(line, "|");
  while (field && count < 7) {
    fields[count++] = field;
    field = strtok(nullptr, "|");
  }
  if ((count != 5 && count != 7) || field != nullptr) {
    printError("expected five fields, or seven fields including model and seed");
    return;
  }
  if ((strcmp(fields[1], "validation") != 0 && strcmp(fields[1], "test") != 0) || !fields[0][0] || !fields[2][0]) {
    printError("invalid provenance fields");
    return;
  }
  float prediction = 0.0f;
  float reference = 0.0f;
  if (!parseBpm(fields[3], prediction) || !parseBpm(fields[4], reference)) {
    printError("BPM must be finite");
    return;
  }
  const bool extended = count == 7;
  if (extended && ((strcmp(fields[5], "E3") && strcmp(fields[5], "E4")) ||
                  (strcmp(fields[6], "42") && strcmp(fields[6], "123") && strcmp(fields[6], "999")))) {
    printError("unsupported model or seed");
    return;
  }
  drawReading(fields, prediction, reference, extended);
  Serial.println("\n--- PPG HEART-RATE REPLAY ---");
  Serial.printf("Window: %s\nSplit: %s | Activity: %s\n", fields[0], fields[1], fields[2]);
  Serial.printf("Host model BPM: %.2f\nECG reference: %.2f\nAbsolute error: %.2f BPM\n",
                prediction, reference, fabsf(prediction - reference));
  Serial.println("Inference platform: host computer; ESP32 simulator handles display only.");
  Serial.printf("ACK|%s|%.2f|%.2f", fields[0], prediction, reference);
  if (extended) Serial.printf("|%s|%s|%s", fields[5], fields[6], fields[1]);
  Serial.println();
}
}  // namespace

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT);
  display.begin();
  display.setRotation(0);
  display.fillScreen(ILI9341_BLACK);
  label(10, 20, "PPG + ACC DEMO", ILI9341_CYAN);
  label(10, 65, "Waiting for host CPU", ILI9341_WHITE, 1);
  label(10, 100, "ESP32 firmware: display / UART", ILI9341_YELLOW, 1);
  Serial.println("PPG-DaLiA recorded replay receiver ready.");
  Serial.println("Send: window_id|split|activity|prediction_bpm|reference_bpm|model|seed");
}

void loop() {
  while (Serial.available() > 0) {
    const char value = static_cast<char>(Serial.read());
    if (value == '\r') continue;
    if (value == '\n') {
      lineBuffer[lineLength] = '\0';
      if (!lineOverflow && lineLength > 0) {
        showReading(lineBuffer);
        digitalWrite(2, !digitalRead(2));
      }
      lineLength = 0;
      lineOverflow = false;
    } else if (lineOverflow) {
      continue;
    } else if (lineLength + 1 < kLineCapacity) {
      lineBuffer[lineLength++] = value;
    } else {
      lineLength = 0;
      lineOverflow = true;
      printError("serial line exceeds 191 bytes");
    }
  }
}
