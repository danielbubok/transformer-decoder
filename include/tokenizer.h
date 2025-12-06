#ifndef TOKENIZER_H
#define TOKENIZER_H

// byte-level tokenizer with 264 vocabulary entries
// ids 0-255: raw bytes
// ids 256-263: special tokens

#define TOK_BYTE_COUNT 256
#define TOK_SYSTEM_OPEN 256
#define TOK_SYSTEM_CLOSE 257
#define TOK_THINK_OPEN 258
#define TOK_THINK_CLOSE 259
#define TOK_PROMPT_OPEN 260
#define TOK_PROMPT_CLOSE 261
#define TOK_ANSWER_OPEN 262
#define TOK_ANSWER_CLOSE 263
#define TOK_VOCAB_SIZE 264

// encode text to token ids
// returns number of tokens written, or -1 on error
// guarantees: encode(decode(tokens)) == tokens for valid tokens
int tokenize(const char *text, int *tokens, int max_tokens);

// decode token ids to text
// returns number of characters written (excluding null terminator)
// guarantees: decode(encode(text)) == text for text without partial special
// tokens
int detokenize(const int *tokens, int n_tokens, char *out, int max_len);

// check if token is a special token
int is_special_token(int token);

// get special token string (returns null for non-special tokens)
const char *get_special_token_str(int token);

// get special token id from string (returns -1 if not found)
int get_special_token_id(const char *str);

#endif
