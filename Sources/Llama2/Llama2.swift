// The Swift Programming Language
// https://docs.swift.org/swift-book
// 
// Swift Argument Parser
// https://swiftpackageindex.com/apple/swift-argument-parser/documentation

import ArgumentParser
import Foundation

// MARK: - Enums

enum Mode: String, CaseIterable, ExpressibleByArgument {
    case generate, chat
    
    static var allValueStrings: [String] {
        allCases.map { $0.rawValue }
    }
}

// MARK: - Engine

final class Llama2Engine {
    private let transformer: Transformer
    private var tokenizer: Tokenizer
    private var sampler: Sampler
    
    init(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler) {
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.sampler = sampler
    }
    
    func generate(prompt: String?, steps: Int) throws -> String {
        let emptyPrompt = "once upon a time"
        let actualPrompt = prompt ?? emptyPrompt
        
        // Encode the prompt into tokens
        let promptTokens = tokenizer.encode(text: actualPrompt, bos: true, eos: false)
        
        guard !promptTokens.isEmpty else {
            throw Llama2Error.invalidParameter("Expected at least 1 prompt token")
        }
        
        // Start the main loop
        var start: TimeInterval = 0 // Used to time our code, only initialized after first iteration
        var next: Int // Will store the next token in the sequence
        var token = promptTokens[0] // Kick off with the first token in the prompt
        var pos = 0 // Position in the sequence
        var output = ""
        
        while pos < steps {
            // Forward the transformer to get logits for the next token
            let logits = transformer.forward(token: token, pos: pos)
            
            // Advance the state machine
            if pos < promptTokens.count - 1 {
                // If we are still processing the input prompt, force the next prompt token
                next = promptTokens[pos + 1]
            } else {
                // Otherwise sample the next token from the logits
                next = sampler.sample(logits: logits)
            }
            pos += 1
            
            // Data-dependent terminating condition: the BOS (=1) token delimits sequences
            if next == 1 { break }
            
            // Print the token as string, decode it with the Tokenizer object
            let piece = tokenizer.decode(prevToken: token, token: next)
            tokenizer.safePrint(piece)
            output += piece
            token = next
            
            // Init the timer here because the first iteration can be slower
            if start == 0 { start = Date().timeIntervalSince1970 }
        }
        
        print("\n")
        
        // Report achieved tok/s (pos-1 because the timer starts after first iteration)
        if pos > 1 {
            let end = Date().timeIntervalSince1970
            let tokPerSec = Double(pos - 1) / (end - start)
            fputs("achieved tok/s: \(tokPerSec)\n", stderr)
        }
        
        return output
    }
    
    func chat(prompt: String?, systemPrompt: String?, steps: Int) throws -> String {
        // Buffers for reading the system prompt and user prompt
        var systemPromptBuffer = ""
        var userPromptBuffer = ""
        var renderedPrompt = ""
        var promptTokens: [Int] = []
        var userIdx = 0
        
        // Start the main loop
        var userTurn = true // User starts
        var next: Int = 0 // Will store the next token in the sequence
        var token: Int = 0 // Stores the current token to feed into the transformer
        var pos = 0 // Position in the sequence
        var output = ""
        
        while pos < steps {
            // When it is the user's turn to contribute tokens to the dialog...
            if userTurn {
                // Get the (optional) system prompt at position 0
                if pos == 0 {
                    // At position 0, the user can also contribute a system prompt
                    if let cliSystemPrompt = systemPrompt {
                        // System prompt was passed in, use it
                        systemPromptBuffer = cliSystemPrompt
                    } else {
                        // System prompt was not passed in, attempt to get it from stdin
                        print("Enter system prompt (optional): ", terminator: "")
                        systemPromptBuffer = readLine() ?? ""
                    }
                }
                
                // Get the user prompt
                if pos == 0, let cliUserPrompt = prompt {
                    // User prompt for position 0 was passed in, use it
                    userPromptBuffer = cliUserPrompt
                } else {
                    // Otherwise get user prompt from stdin
                    print("User: ", terminator: "")
                    userPromptBuffer = readLine() ?? ""
                }
                
                // Render user/system prompts into the Llama 2 Chat schema
                if pos == 0 && !systemPromptBuffer.isEmpty {
                    let systemTemplate = "[INST] <<SYS>>\n\(systemPromptBuffer)\n<</SYS>>\n\n\(userPromptBuffer) [/INST]"
                    renderedPrompt = systemTemplate
                } else {
                    let userTemplate = "[INST] \(userPromptBuffer) [/INST]"
                    renderedPrompt = userTemplate
                }
                
                // Encode the rendered prompt into tokens
                promptTokens = tokenizer.encode(text: renderedPrompt, bos: true, eos: false)
                userIdx = 0 // Reset the user index
                userTurn = false
                print("Assistant: ", terminator: "")
            }
            
            // Determine the token to pass into the transformer next
            if userIdx < promptTokens.count {
                // If we are still processing the input prompt, force the next prompt token
                token = promptTokens[userIdx]
                userIdx += 1
            } else {
                // Otherwise use the next token sampled from previous turn
                token = next
            }
            
            // EOS (=2) token ends the Assistant turn
            if token == 2 { userTurn = true }
            
            // Forward the transformer to get logits for the next token
            let logits = transformer.forward(token: token, pos: pos)
            next = sampler.sample(logits: logits)
            pos += 1
            
            if userIdx >= promptTokens.count && next != 2 {
                // The Assistant is responding, so print its output
                let piece = tokenizer.decode(prevToken: token, token: next)
                tokenizer.safePrint(piece)
                output += piece
            }
            
            if next == 2 { print("\n") }
        }
        
        print("\n")
        return output
    }
}

// MARK: - Parameter Validation

struct GenerationParameters {
    let temperature: Float
    let topP: Float
    let seed: UInt64
    let steps: Int
    let prompt: String?
    let systemPrompt: String?
    
    init(
        temperature: Float,
        topP: Float,
        seed: UInt64,
        steps: Int,
        prompt: String?,
        systemPrompt: String?
    ) throws {
        // Validate temperature
        guard temperature >= 0.0 else {
            throw Llama2Error.invalidTemperature(temperature)
        }
        
        // Validate topp
        guard topP >= 0.0 && topP <= 1.0 else {
            throw Llama2Error.invalidTopP(topP)
        }
        
        // Validate steps
        guard steps >= 0 else {
            throw Llama2Error.invalidSteps(steps)
        }
        
        self.temperature = temperature
        self.topP = topP
        self.seed = seed == 0 ? UInt64(Date().timeIntervalSince1970) : seed
        self.steps = steps
        self.prompt = prompt
        self.systemPrompt = systemPrompt
    }
}

// MARK: - CLI

@main
struct Llama2: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "A Swift implementation of Llama2 text generation",
        discussion: """
        Example: Llama2 model.bin -n 256 -i "Once upon a time"
        """
    )
    
    @Argument(help: "Path to the model checkpoint file (e.g., out/model.bin)")
    var checkpointPath: String = Bundle.module.path(forResource: "Resources/stories260K", ofType: "bin") ?? ""
    
    @Option(name: .customShort("t"), help: "Temperature in [0,inf], default 1.0")
    var temperature: Float = 1.0
    
    @Option(name: .customShort("p"), help: "P value in top-p (nucleus) sampling in [0,1], default 0.9")
    var topP: Float = 0.9
    
    @Option(name: .customShort("s"), help: "Random seed, default time(NULL)")
    var seed: UInt64 = 0
    
    @Option(name: .customShort("n"), help: "Number of steps to run for, default 256. 0 = max_seq_len")
    var steps: Int = 256
    
    @Option(name: .customShort("i"), help: "Input prompt")
    var prompt: String?
    
    @Option(name: .customShort("z"), help: "Optional path to custom tokenizer")
    var tokenizerPath: String = Bundle.module.path(forResource: "Resources/tokenizer", ofType: "json") ?? ""
    
    @Option(name: .customShort("m"), help: "Mode: generate|chat, default: generate")
    var mode: Mode = .generate
    
    @Option(name: .customShort("y"), help: "Optional system prompt in chat mode")
    var systemPrompt: String?
    
    func run() throws {
        // Validate and prepare parameters
        let params = try GenerationParameters(
            temperature: temperature,
            topP: topP,
            seed: seed,
            steps: steps,
            prompt: prompt,
            systemPrompt: systemPrompt
        )
                
        // Create components with actual model data
        let tokenizer = try Tokenizer(tokenizerPath: tokenizerPath)
        let transformer = try Transformer(checkpointPath: checkpointPath)
        let sampler = Sampler(temperature: params.temperature, topp: params.topP, seed: params.seed)
        
        // Sanity check 1: Verify vocabulary sizes match
        let tokenizerVocabSize = tokenizer.vocabularySize
        let modelVocabSize = Int(transformer.config.vocabSize)
        
        guard tokenizerVocabSize == modelVocabSize else {
            throw Llama2Error.invalidParameter("Vocabulary size mismatch: tokenizer has \(tokenizerVocabSize) tokens, model expects \(modelVocabSize) tokens")
        }
        
        // Sanity check 2: Verify token embedding table size is correct
        let expectedEmbeddingSize = modelVocabSize * Int(transformer.config.dim)
        let actualEmbeddingSize = transformer.weights.tokenEmbeddingTable.count
        
        guard actualEmbeddingSize == expectedEmbeddingSize else {
            throw Llama2Error.invalidParameter("Token embedding table size mismatch: expected \(expectedEmbeddingSize), got \(actualEmbeddingSize)")
        }
        
        // Sanity check 3: Verify model configuration is reasonable
        guard transformer.config.dim > 0 else {
            throw Llama2Error.invalidParameter("Invalid model dimension: \(transformer.config.dim)")
        }
        
        guard transformer.config.numLayers > 0 else {
            throw Llama2Error.invalidParameter("Invalid number of layers: \(transformer.config.numLayers)")
        }
        
        guard transformer.config.seqLen > 0 else {
            throw Llama2Error.invalidParameter("Invalid sequence length: \(transformer.config.seqLen)")
        }
        
        // Print configuration summary for debugging
        fputs("Configuration:\n", stderr)
        fputs("  Model vocab size: \(modelVocabSize)\n", stderr)
        fputs("  Tokenizer vocab size: \(tokenizerVocabSize)\n", stderr)
        fputs("  Model dimension: \(transformer.config.dim)\n", stderr)
        fputs("  Number of layers: \(transformer.config.numLayers)\n", stderr)
        fputs("  Sequence length: \(transformer.config.seqLen)\n", stderr)
        fputs("  Temperature: \(params.temperature)\n", stderr)
        fputs("  Top-p: \(params.topP)\n", stderr)
        fputs("  Steps: \(params.steps)\n", stderr)
        fputs("  Mode: \(mode)\n", stderr)
        if let prompt = params.prompt {
            fputs("  Prompt: \"\(prompt)\"\n", stderr)
        }
        fputs("\n", stderr)
        
        // Create engine
        let engine = Llama2Engine(
            transformer: transformer,
            tokenizer: tokenizer,
            sampler: sampler
        )
        
        // Determine actual steps
        let actualSteps = params.steps == 0 ? Int(transformer.config.seqLen) : params.steps
        
        // Generate output
        let output: String
        switch mode {
        case .generate:
            output = try engine.generate(prompt: params.prompt, steps: actualSteps)
        case .chat:
            output = try engine.chat(prompt: params.prompt, systemPrompt: params.systemPrompt, steps: actualSteps)
        }
        
        print(output)         
    }
}

/*
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 
 // ----------------------------------------------------------------------------
 // generation loop

 void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
     char *empty_prompt = "";
     if (prompt == NULL) { prompt = empty_prompt; }

     // encode the (string) prompt into tokens sequence
     int num_prompt_tokens = 0;
     int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
     encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
     if (num_prompt_tokens < 1) {
         fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
         exit(EXIT_FAILURE);
     }

     // start the main loop
     long start = 0;  // used to time our code, only initialized after first iteration
     int next;        // will store the next token in the sequence
     int token = prompt_tokens[0]; // kick off with the first token in the prompt
     int pos = 0;     // position in the sequence
     while (pos < steps) {

         // forward the transformer to get logits for the next token
         float* logits = forward(transformer, token, pos);

         // advance the state machine
         if (pos < num_prompt_tokens - 1) {
             // if we are still processing the input prompt, force the next prompt token
             next = prompt_tokens[pos + 1];
         } else {
             // otherwise sample the next token from the logits
             next = sample(sampler, logits);
         }
         pos++;

         // data-dependent terminating condition: the BOS (=1) token delimits sequences
         if (next == 1) { break; }

         // print the token as string, decode it with the Tokenizer object
         char* piece = decode(tokenizer, token, next);
         safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
         fflush(stdout);
         token = next;

         // init the timer here because the first iteration can be slower
         if (start == 0) { start = time_in_ms(); }
     }
     printf("\n");

     // report achieved tok/s (pos-1 because the timer starts after first iteration)
     if (pos > 1) {
         long end = time_in_ms();
         fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
     }

     free(prompt_tokens);
 }

 void read_stdin(const char* guide, char* buffer, size_t bufsize) {
     // read a line from stdin, up to but not including \n
     printf("%s", guide);
     if (fgets(buffer, bufsize, stdin) != NULL) {
         size_t len = strlen(buffer);
         if (len > 0 && buffer[len - 1] == '\n') {
             buffer[len - 1] = '\0'; // strip newline
         }
     }
 }

 // ----------------------------------------------------------------------------
 // chat loop
 // I manually inspected the tokens for a few chat conversations compared to
 // python reference and that seemed ok, but this was not thoroughly tested and
 // is not safely implemented, it's more a proof of concept atm.

 void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
           char *cli_user_prompt, char *cli_system_prompt, int steps) {

     // buffers for reading the system prompt and user prompt from stdin
     // you'll notice they are soomewhat haphazardly and unsafely set atm
     char system_prompt[512];
     char user_prompt[512];
     char rendered_prompt[1152];
     int num_prompt_tokens = 0;
     int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
     int user_idx;

     // start the main loop
     int8_t user_turn = 1; // user starts
     int next;        // will store the next token in the sequence
     int token;       // stores the current token to feed into the transformer
     int prev_token;
     int pos = 0;     // position in the sequence
     while (pos < steps) {

         // when it is the user's turn to contribute tokens to the dialog...
         if (user_turn) {
             // get the (optional) system prompt at position 0
             if (pos == 0) {
                 // at position 0, the user can also contribute a system prompt
                 if (cli_system_prompt == NULL) {
                     // system prompt was not passed in, attempt to get it from stdin
                     read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                 } else {
                     // system prompt was passed in, use it
                     strcpy(system_prompt, cli_system_prompt);
                 }
             }
             // get the user prompt
             if (pos == 0 && cli_user_prompt != NULL) {
                 // user prompt for position 0 was passed in, use it
                 strcpy(user_prompt, cli_user_prompt);
             } else {
                 // otherwise get user prompt from stdin
                 read_stdin("User: ", user_prompt, sizeof(user_prompt));
             }
             // render user/system prompts into the Llama 2 Chat schema
             if (pos == 0 && system_prompt[0] != '\0') {
                 char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                 sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
             } else {
                 char user_template[] = "[INST] %s [/INST]";
                 sprintf(rendered_prompt, user_template, user_prompt);
             }
             // encode the rendered prompt into tokens
             encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
             user_idx = 0; // reset the user index
             user_turn = 0;
             printf("Assistant: ");
         }

         // determine the token to pass into the transformer next
         if (user_idx < num_prompt_tokens) {
             // if we are still processing the input prompt, force the next prompt token
             token = prompt_tokens[user_idx++];
         } else {
             // otherwise use the next token sampled from previous turn
             token = next;
         }
         // EOS (=2) token ends the Assistant turn
         if (token == 2) { user_turn = 1; }

         // forward the transformer to get logits for the next token
         float* logits = forward(transformer, token, pos);
         next = sample(sampler, logits);
         pos++;

         if (user_idx >= num_prompt_tokens && next != 2) {
             // the Assistant is responding, so print its output
             char* piece = decode(tokenizer, token, next);
             safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
             fflush(stdout);
         }
         if (next == 2) { printf("\n"); }
     }
     printf("\n");
     free(prompt_tokens);
 }

 int main(int argc, char *argv[]) {

     // default parameters
     char *checkpoint_path = NULL;  // e.g. out/model.bin
     char *tokenizer_path = "tokenizer.bin";
     float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
     float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
     int steps = 256;            // number of steps to run for
     char *prompt = NULL;        // prompt string
     unsigned long long rng_seed = 0; // seed rng with time by default
     char *mode = "generate";    // generate|chat
     char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

     // poor man's C argparse so we can override the defaults above from the command line
     if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
     for (int i = 2; i < argc; i+=2) {
         // do some basic validation
         if (i + 1 >= argc) { error_usage(); } // must have arg after flag
         if (argv[i][0] != '-') { error_usage(); } // must start with dash
         if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
         // read in the args
         if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
         else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
         else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
         else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
         else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
         else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
         else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
         else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
         else { error_usage(); }
     }

     // parameter validation/overrides
     if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
     if (temperature < 0.0) temperature = 0.0;
     if (topp < 0.0 || 1.0 < topp) topp = 0.9;
     if (steps < 0) steps = 0;

     // build the Transformer via the model .bin file
     Transformer transformer;
     build_transformer(&transformer, checkpoint_path);
     if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

     // build the Tokenizer via the tokenizer .bin file
     Tokenizer tokenizer;
     build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

     // build the Sampler
     Sampler sampler;
     build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

     // run!
     if (strcmp(mode, "generate") == 0) {
         generate(&transformer, &tokenizer, &sampler, prompt, steps);
     } else if (strcmp(mode, "chat") == 0) {
         chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
     } else {
         fprintf(stderr, "unknown mode: %s\n", mode);
         error_usage();
     }

     // memory and file handles cleanup
     free_sampler(&sampler);
     free_tokenizer(&tokenizer);
     free_transformer(&transformer);
     return 0;
 }

 */
