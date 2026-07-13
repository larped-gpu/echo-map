/*
 * EchoMap: robot base controller for ESP32
 *
 * Streams odometry CSV at 20 Hz:
 *   timestamp_ms,left_ticks,right_ticks,heading_deg,servo_deg
 *
 * Hardware:
 *   L298N motor driver -> GPIO 25,26 (left) / 27,14 (right)
 *   Wheel encoders     -> GPIO 18 (left) / 19 (right) with interrupts
 *   MPU-6050 IMU       -> I2C SDA=21, SCL=22
 *   SG90 servo (pan)   -> GPIO 13
 */

#define USE_WIFI 0

#include <Wire.h>

#if USE_WIFI
#include <WiFi.h>
#include <WiFiUdp.h>
const char *WIFI_SSID = "YOUR_SSID";
const char *WIFI_PASS = "YOUR_PASSWORD";
const uint16_t UDP_PORT = 5006;
const char *UDP_HOST = "192.168.1.100";
WiFiUDP udp;
#endif

// Motor pins (L298N)
static const int MOTOR_L_FWD = 25;
static const int MOTOR_L_BWD = 26;
static const int MOTOR_R_FWD = 27;
static const int MOTOR_R_BWD = 14;

// Encoder pins
static const int ENC_L_PIN = 18;
static const int ENC_R_PIN = 19;

// Servo
static const int SERVO_PIN = 13;
static const int SERVO_MIN_US = 500;
static const int SERVO_MAX_US = 2400;

// IMU (MPU-6050)
static const int IMU_SDA = 21;
static const int IMU_SCL = 22;

static const int TICKS_PER_REV = 20;
static const float WHEEL_DIAMETER_M = 0.065;
static const float WHEEL_BASE_M = 0.14;

volatile long leftTicks = 0;
volatile long rightTicks = 0;

float headingDeg = 0.0;
float servoDeg = 0.0;

unsigned long lastOdometryMs = 0;
static const unsigned long ODOMETRY_INTERVAL_MS = 50; // 20 Hz

void IRAM_ATTR onLeftEncoder() {
  leftTicks += digitalRead(ENC_L_PIN) == digitalRead(ENC_R_PIN) ? 1 : -1;
}

void IRAM_ATTR onRightEncoder() {
  rightTicks += digitalRead(ENC_R_PIN) == digitalRead(ENC_L_PIN) ? 1 : -1;
}

void setupMotors() {
  pinMode(MOTOR_L_FWD, OUTPUT);
  pinMode(MOTOR_L_BWD, OUTPUT);
  pinMode(MOTOR_R_FWD, OUTPUT);
  pinMode(MOTOR_R_BWD, OUTPUT);
  digitalWrite(MOTOR_L_FWD, LOW);
  digitalWrite(MOTOR_L_BWD, LOW);
  digitalWrite(MOTOR_R_FWD, LOW);
  digitalWrite(MOTOR_R_BWD, LOW);
}

void setupEncoders() {
  pinMode(ENC_L_PIN, INPUT_PULLUP);
  pinMode(ENC_R_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_PIN), onLeftEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_R_PIN), onRightEncoder, CHANGE);
}

void setupServo() {
  pinMode(SERVO_PIN, OUTPUT);
  setServoAngle(0.0);
}

void setServoAngle(float angleDeg) {
  angleDeg = constrain(angleDeg, -90.0, 90.0);
  servoDeg = angleDeg;
  float us = map((long)angleDeg, -90, 90, SERVO_MIN_US, SERVO_MAX_US);
  // Simple servo pulse (not using Servo library for pin flexibility)
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds((int)us);
  digitalWrite(SERVO_PIN, LOW);
  delayMicroseconds(20000 - (int)us);
}

void setMotorSpeed(int leftPct, int rightPct) {
  // Left motor
  if (leftPct >= 0) {
    analogWrite(MOTOR_L_FWD, leftPct);
    analogWrite(MOTOR_L_BWD, 0);
  } else {
    analogWrite(MOTOR_L_FWD, 0);
    analogWrite(MOTOR_L_BWD, -leftPct);
  }
  // Right motor
  if (rightPct >= 0) {
    analogWrite(MOTOR_R_FWD, rightPct);
    analogWrite(MOTOR_R_BWD, 0);
  } else {
    analogWrite(MOTOR_R_FWD, 0);
    analogWrite(MOTOR_R_BWD, -rightPct);
  }
}

void readImuHeading() {
  // Placeholder: read MPU-6050 gyro Z for heading integration
  // Full I2C driver to be added when IMU arrives
  // For now, heading is set externally or drifts slowly
}

void emitOdometryCsv(unsigned long timestampMs) {
  noInterrupts();
  long lt = leftTicks;
  long rt = rightTicks;
  interrupts();

  Serial.print(timestampMs);
  Serial.print(',');
  Serial.print(lt);
  Serial.print(',');
  Serial.print(rt);
  Serial.print(',');
  Serial.print(headingDeg, 1);
  Serial.print(',');
  Serial.println(servoDeg, 1);
}

#if USE_WIFI
void emitOdometryUdp(unsigned long timestampMs) {
  noInterrupts();
  long lt = leftTicks;
  long rt = rightTicks;
  interrupts();

  char line[64];
  int n = snprintf(line, sizeof(line), "%lu,%ld,%ld,%.1f,%.1f",
                   timestampMs, lt, rt, headingDeg, servoDeg);
  udp.beginPacket(UDP_HOST, UDP_PORT);
  udp.write((const uint8_t *)line, n);
  udp.endPacket();
}
#endif

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  setupMotors();
  setupEncoders();
  setupServo();

  Wire.begin(IMU_SDA, IMU_SCL);

#if USE_WIFI
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  udp.begin(UDP_PORT);
  Serial.println("# wifi_connected");
#endif

  Serial.println("# echomap_v1");
  Serial.println("# format: timestamp_ms,left_ticks,right_ticks,heading_deg,servo_deg");
}

void loop() {
  unsigned long now = millis();

  readImuHeading();

  if (now - lastOdometryMs >= ODOMETRY_INTERVAL_MS) {
    lastOdometryMs = now;
    emitOdometryCsv(now);
#if USE_WIFI
    emitOdometryUdp(now);
#endif
  }

  // Serial command interface: "S,<angle>" to set servo, "M,<l>,<r>" for motors
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("S,")) {
      float angle = cmd.substring(2).toFloat();
      setServoAngle(angle);
    } else if (cmd.startsWith("M,")) {
      int comma = cmd.indexOf(',', 2);
      int left = cmd.substring(2, comma).toInt();
      int right = cmd.substring(comma + 1).toInt();
      setMotorSpeed(left, right);
    } else if (cmd == "STOP") {
      setMotorSpeed(0, 0);
    }
  }
}
