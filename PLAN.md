# Plan

- [x] Inspect `slvDev/esp32-ai`, its license, and the connected XIAO.
- [x] Create the private `kortexa-ai/esp32-mind` repository.
- [x] Size and train a TinyStories PLE model for 8 MB flash.
- [x] Add a USB serial prompt/response protocol and host client.
- [x] Build and flash the XIAO ESP32-S3.
- [x] Send a prompt over USB and capture an on-device response.
- [x] Document, validate, commit to `main`, and push.

## Constraints

- The upstream source is MIT-licensed and remains untouched.
- The connected XIAO has 8 MB flash and 8 MB PSRAM.
- Model artifacts and training data stay gitignored.
- `kortexa-ai/esp32-mind` follows Sparta rules: direct commits to `main`, no PR.
