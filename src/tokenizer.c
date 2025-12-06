#include "tokenizer.h"
#include <string.h>

typedef struct {
  const char *text;
  int token;
  int len;
} SpecialToken;

static const SpecialToken SPECIAL_TOKENS[] = {
    {"<system>", TOK_SYSTEM_OPEN, 8}, {"</system>", TOK_SYSTEM_CLOSE, 9},
    {"<think>", TOK_THINK_OPEN, 7},   {"</think>", TOK_THINK_CLOSE, 8},
    {"<prompt>", TOK_PROMPT_OPEN, 8}, {"</prompt>", TOK_PROMPT_CLOSE, 9},
    {"<answer>", TOK_ANSWER_OPEN, 8}, {"</answer>", TOK_ANSWER_CLOSE, 9},
};

static const int N_SPECIAL = sizeof(SPECIAL_TOKENS) / sizeof(SPECIAL_TOKENS[0]);

int is_special_token(int token) {
  return token >= TOK_SYSTEM_OPEN && token <= TOK_ANSWER_CLOSE;
}

const char *get_special_token_str(int token) {
  for (int i = 0; i < N_SPECIAL; i++) {
    if (SPECIAL_TOKENS[i].token == token) {
      return SPECIAL_TOKENS[i].text;
    }
  }
  return (void *)0;
}

int get_special_token_id(const char *str) {
  for (int i = 0; i < N_SPECIAL; i++) {
    if (strcmp(str, SPECIAL_TOKENS[i].text) == 0) {
      return SPECIAL_TOKENS[i].token;
    }
  }
  return -1;
}

static int try_special(const char *text, int *token) {
  for (int i = 0; i < N_SPECIAL; i++) {
    if (strncmp(text, SPECIAL_TOKENS[i].text, (size_t)SPECIAL_TOKENS[i].len) ==
        0) {
      *token = SPECIAL_TOKENS[i].token;
      return SPECIAL_TOKENS[i].len;
    }
  }
  return 0;
}

int tokenize(const char *text, int *tokens, int max_tokens) {
  if (!text || !tokens || max_tokens <= 0)
    return -1;

  int n = 0;
  int i = 0;
  int len = (int)strlen(text);

  while (i < len && n < max_tokens) {
    int tok;
    int consumed = try_special(text + i, &tok);

    if (consumed > 0) {
      tokens[n++] = tok;
      i += consumed;
    } else {
      // byte token
      tokens[n++] = (unsigned char)text[i];
      i++;
    }
  }

  return n;
}

int detokenize(const int *tokens, int n_tokens, char *out, int max_len) {
  if (!tokens || !out || max_len <= 0)
    return -1;

  int pos = 0;

  for (int i = 0; i < n_tokens && pos < max_len - 1; i++) {
    int tok = tokens[i];

    if (tok >= 0 && tok < TOK_BYTE_COUNT) {
      // byte token
      out[pos++] = (char)tok;
    } else if (is_special_token(tok)) {
      // special token
      const char *s = get_special_token_str(tok);
      if (s) {
        int slen = (int)strlen(s);
        for (int j = 0; j < slen && pos < max_len - 1; j++) {
          out[pos++] = s[j];
        }
      }
    }
    // invalid tokens are silently ignored
  }

  out[pos] = '\0';
  return pos;
}
