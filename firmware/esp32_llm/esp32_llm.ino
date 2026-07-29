// TinyStories PLE inference for the Seeed Studio XIAO ESP32-S3.
//
// The quantized model lives in a memory-mapped flash partition. Prompts arrive
// as token IDs over USB serial because BPE tokenization stays on the host.

#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"

#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "vocab.h"

static const int MAX_PROMPT_TOKENS = 96;
static const int MAX_GENERATE_TOKENS = 128;
static const int COMMAND_BUFFER_SIZE = 1024;
static const int EOT_TOKEN = 0;

Model model;
Scratch scratch;

static bool model_ready = false;
static esp_partition_mmap_handle_t model_mmap_handle;

// ---- int8 output head -------------------------------------------------------
// The output head is scanned for every generated token. Unpacking it once into
// PSRAM makes the hot path an int8 x int8 dot product and lets both cores split
// the vocabulary rows.
static int8_t *head_w8 = NULL;
static float *head_scale8 = NULL;
static int head_rows;
static int head_cols;
static int8_t head_actq[128];
static float head_acts;

static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) {
    acc += (int32_t)a[i] * (int32_t)b[i];
  }
  return acc;
}

static void head_rows_range(float *y, int row_begin, int row_end) {
  for (int row = row_begin; row < row_end; row++) {
    y[row] =
        (float)dot_i8(head_actq, head_w8 + (size_t)row * head_cols, head_cols) *
        head_scale8[row] * head_acts;
  }
}

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

static void head_matvec_int8(const QT *, const float *x, float *y) {
  quantize_act(x, head_cols, head_actq, &head_acts);
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *psram_alloc(size_t bytes) {
  void *ptr = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM);
  if (!ptr) {
    Serial.printf("MIND FATAL psram_alloc bytes=%u\n", (unsigned)bytes);
    while (true) {
      delay(1000);
    }
  }
  return ptr;
}

static void stage_head_int8(QT *tensor) {
  head_rows = tensor->rows;
  head_cols = tensor->cols;
  head_w8 = (int8_t *)psram_alloc((size_t)head_rows * head_cols);
  head_scale8 = (float *)psram_alloc((size_t)head_rows * sizeof(float));

  for (int row_index = 0; row_index < head_rows; row_index++) {
    const uint8_t *row =
        tensor->codes + (size_t)row_index * tensor->row_bytes;
    int8_t *destination =
        head_w8 + (size_t)row_index * head_cols;
    for (int column = 0; column < head_cols; column++) {
      uint8_t byte = row[column >> 1];
      int code = (column & 1) ? (byte >> 4) : (byte & 0x0f);
      destination[column] = (int8_t)(code - 8);
    }
    head_scale8[row_index] =
        half2float(tensor->scales[(size_t)row_index * tensor->n_groups]);
  }
}

static void allocate_scratch() {
  Cfg *config = &model.c;
  int dim = config->dim;
  int layers = config->n_layers;
  int ple_dim = config->ple_dim;
  int ffn = config->ffn;
  int vocab = config->vocab;
  int sequence = config->seq_len;

  scratch.x = (float *)psram_alloc(dim * sizeof(float));
  scratch.h = (float *)psram_alloc((ffn > dim ? ffn : dim) * sizeof(float));
  scratch.qkv = (float *)psram_alloc(3 * dim * sizeof(float));
  scratch.att = (float *)psram_alloc(dim * sizeof(float));
  scratch.g1 = (float *)psram_alloc(ffn * sizeof(float));
  scratch.g2 =
      (float *)psram_alloc((ple_dim > ffn ? ple_dim : ffn) * sizeof(float));
  scratch.ple = (float *)psram_alloc(layers * ple_dim * sizeof(float));
  scratch.tmpP = (float *)psram_alloc(layers * ple_dim * sizeof(float));
  scratch.trow = (float *)psram_alloc(layers * ple_dim * sizeof(float));
  scratch.logits = (float *)psram_alloc(vocab * sizeof(float));
  scratch.scores = (float *)psram_alloc(sequence * sizeof(float));
  scratch.kcache =
      (float *)psram_alloc((size_t)layers * sequence * dim * sizeof(float));
  scratch.vcache =
      (float *)psram_alloc((size_t)layers * sequence * dim * sizeof(float));
}

static bool load_model() {
  const esp_partition_t *partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!partition) {
    Serial.println("MIND ERROR model_partition_missing");
    return false;
  }

  const void *base = NULL;
  esp_err_t error = esp_partition_mmap(
      partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &base,
      &model_mmap_handle);
  if (error != ESP_OK) {
    Serial.printf("MIND ERROR model_mmap code=%d\n", error);
    return false;
  }
  if (llm_load((const uint8_t *)base, &model)) {
    Serial.println("MIND ERROR bad_model_magic");
    return false;
  }
  if (VOCAB_N > model.c.vocab) {
    Serial.printf("MIND ERROR vocab_mismatch asset=%d model=%d\n", VOCAB_N,
                  model.c.vocab);
    return false;
  }
  if (model.c.dim > (int)sizeof(head_actq)) {
    Serial.printf("MIND ERROR model_dim_too_large dim=%d\n", model.c.dim);
    return false;
  }

  // Tokenizers can finish with fewer entries than the configured padded
  // vocabulary. Do not stage or score unreachable rows.
  model.tok_emb.rows = VOCAB_N;
  stage_head_int8(&model.tok_emb);

  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    Serial.println("MIND ERROR head_worker_create");
    return false;
  }
  model.head_matvec = head_matvec_int8;
  allocate_scratch();
  return true;
}

static void print_info() {
  Serial.printf(
      "MIND INFO vocab=%d dim=%d layers=%d heads=%d ffn=%d ple=%d context=%d "
      "psram_free_kb=%u\n",
      VOCAB_N, model.c.dim, model.c.n_layers, model.c.n_heads, model.c.ffn,
      model.c.ple_dim, model.c.seq_len,
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
}

static void emit_token(int token) {
  if (token < 0 || token >= VOCAB_N) {
    return;
  }
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[token];
  int length = VOCAB_OFF[token + 1] - VOCAB_OFF[token];
  Serial.write(bytes, length);
}

static int best_token() {
  int best = 0;
  float best_value = -1e30f;
  for (int token = 0; token < VOCAB_N; token++) {
    if (scratch.logits[token] > best_value) {
      best_value = scratch.logits[token];
      best = token;
    }
  }
  return best;
}

static void generate(const int *prompt, int prompt_tokens, int max_new_tokens) {
  int position = 0;
  for (int i = 0; i < prompt_tokens; i++) {
    llm_forward(&model, prompt[i], position++, &scratch);
  }

  llm_profile_reset(&scratch);
  Serial.printf("MIND BEGIN prompt_tokens=%d max_new_tokens=%d\n", prompt_tokens,
                max_new_tokens);

  int generated = 0;
  int64_t started = esp_timer_get_time();
  for (; generated < max_new_tokens && position < model.c.seq_len; generated++) {
    int token = best_token();
    if (token == EOT_TOKEN) {
      break;
    }
    emit_token(token);
    llm_forward(&model, token, position++, &scratch);
    if ((generated & 7) == 0) {
      delay(0);
    }
  }
  int64_t elapsed_us = esp_timer_get_time() - started;
  float tokens_per_second =
      elapsed_us > 0 ? generated * 1000000.0f / elapsed_us : 0.0f;

  Serial.printf("\nMIND END tokens=%d seconds=%.2f tok_s=%.2f\n", generated,
                elapsed_us / 1000000.0f, tokens_per_second);
  Serial.println("MIND READY");
  Serial.flush();
}

static void handle_generate(char *save_pointer) {
  if (!model_ready) {
    Serial.println("MIND ERROR model_not_ready");
    return;
  }

  char *value = strtok_r(NULL, " ", &save_pointer);
  if (!value) {
    Serial.println("MIND ERROR usage_generate");
    return;
  }
  int requested = atoi(value);
  if (requested < 1 || requested > MAX_GENERATE_TOKENS) {
    Serial.printf("MIND ERROR max_new_tokens range=1..%d\n",
                  MAX_GENERATE_TOKENS);
    return;
  }

  int prompt[MAX_PROMPT_TOKENS];
  int count = 0;
  while ((value = strtok_r(NULL, " ", &save_pointer)) != NULL) {
    if (count == MAX_PROMPT_TOKENS) {
      Serial.printf("MIND ERROR prompt_too_long max=%d\n", MAX_PROMPT_TOKENS);
      return;
    }
    char *end = NULL;
    long token = strtol(value, &end, 10);
    if (!end || *end != '\0' || token < 0 || token >= VOCAB_N) {
      Serial.printf("MIND ERROR invalid_token value=%s\n", value);
      return;
    }
    prompt[count++] = (int)token;
  }
  if (count == 0) {
    Serial.println("MIND ERROR empty_prompt");
    return;
  }
  if (count + requested > model.c.seq_len) {
    requested = model.c.seq_len - count;
  }
  generate(prompt, count, requested);
}

static void handle_command(char *line) {
  char *save_pointer = NULL;
  char *command = strtok_r(line, " ", &save_pointer);
  if (!command) {
    return;
  }
  if (strcmp(command, "PING") == 0) {
    Serial.println("MIND PONG");
  } else if (strcmp(command, "INFO") == 0) {
    if (model_ready) {
      print_info();
    } else {
      Serial.println("MIND ERROR model_not_ready");
    }
  } else if (strcmp(command, "GENERATE") == 0) {
    handle_generate(save_pointer);
  } else {
    Serial.printf("MIND ERROR unknown_command value=%s\n", command);
  }
}

void setup() {
  Serial.begin(115200);
  delay(750);
  Serial.println("\nMIND BOOT esp32-mind");
  model_ready = load_model();
  if (model_ready) {
    print_info();
    Serial.println("MIND READY");
  }
}

void loop() {
  static char command_buffer[COMMAND_BUFFER_SIZE];
  static int command_length = 0;

  while (Serial.available()) {
    int incoming = Serial.read();
    if (incoming < 0 || incoming == '\r') {
      continue;
    }
    if (incoming == '\n') {
      command_buffer[command_length] = '\0';
      handle_command(command_buffer);
      command_length = 0;
      continue;
    }
    if (command_length < COMMAND_BUFFER_SIZE - 1) {
      command_buffer[command_length++] = (char)incoming;
    } else {
      command_length = 0;
      Serial.println("MIND ERROR command_too_long");
    }
  }
  delay(1);
}
