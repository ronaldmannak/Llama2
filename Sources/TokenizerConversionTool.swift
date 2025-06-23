//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/23/25.
//

import Foundation


/*
 
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 
 
 void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
     // i should have written the vocab_size into the tokenizer file... sigh
     t->vocab_size = vocab_size;
     // malloc space to hold the scores and the strings
     t->vocab = (char**)malloc(vocab_size * sizeof(char*));
     t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
     t->sorted_vocab = NULL; // initialized lazily
     for (int i = 0; i < 256; i++) {
         t->byte_pieces[i * 2] = (unsigned char)i;
         t->byte_pieces[i * 2 + 1] = '\0';
     }
     // read in the file
     FILE *file = fopen(tokenizer_path, "rb");
     if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
     if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
     int len;
     for (int i = 0; i < vocab_size; i++) {
         if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
         if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
         t->vocab[i] = (char *)malloc(len + 1);
         if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
         t->vocab[i][len] = '\0'; // add the string terminating token
     }
     fclose(file);
 }

 void free_tokenizer(Tokenizer* t) {
     for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
     free(t->vocab);
     free(t->vocab_scores);
     free(t->sorted_vocab);
 }

 char* decode(Tokenizer* t, int prev_token, int token) {
     char *piece = t->vocab[token];
     // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
     if (prev_token == 1 && piece[0] == ' ') { piece++; }
     // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
     // parse this and convert and return the actual byte
     unsigned char byte_val;
     if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
         piece = (char*)t->byte_pieces + byte_val * 2;
     }
     return piece;
 }
 */
