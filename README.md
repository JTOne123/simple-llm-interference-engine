# Simple C# LLM inference engine

This project is the smallest practical C# program that runs the supplied
Qwen 2.5 0.5B Instruct GGUF model:

```text
D:\LLMModels\qwen2.5-0.5b-instruct-fp16.gguf
```

The engine uses only the .NET standard library: no LLamaSharp, `llama.cpp`, or
runtime NuGet package. The code directly reads GGUF and implements the Qwen
transformer equations. The separate test project uses MSTest. The engine is
intentionally slow and simple.

## Run it

```powershell
dotnet run --project src\SimpleLlmInference.Console
```

Ask another question by passing the model path and question:

```powershell
dotnet run --project src\SimpleLlmInference.Console -- `
  "D:\LLMModels\qwen2.5-0.5b-instruct-fp16.gguf" `
  "What is the capital of France?"
```

Run the MSTest suite:

```powershell
dotnet test
```

The main test asks, “What is the capital of Ukraine?” and checks that the model
answers “Kyiv”. This is an integration test because it loads and runs the real
1.18 GB model.

## How it works, in plain language

1. **Load the model.** `SimpleInferenceEngine` opens the GGUF file. The file
   contains Qwen's vocabulary, settings, and learned numeric weights.
2. **Turn the question into tokens.** `QwenTokenizer` implements byte-pair
   encoding and Qwen's chat format. Tokens are small text pieces represented by
   numbers.
3. **Use attention.** Inside every transformer layer, Qwen compares each token
   with earlier tokens. `KvCache` remembers their key/value vectors so
   `QwenModel.Attention` can decide which earlier words matter.
4. **Produce logits.** The model gives every possible next token a score called
   a logit. A larger score means “this token is a better next choice”.
5. **Choose greedily.** `ArgMax` takes the token with the highest score. There
   is no randomness, which keeps the example and tests simple.
6. **Repeat.** The chosen token is fed back into the model. Generation stops at
   Qwen's end marker or after the requested token limit.

## What each function does

- `GgufReader` reads metadata and converts FP16 tensors into normal C# floats.
- `QwenTokenizer` changes text to token IDs and token IDs back to text.
- `TransformerMath` contains matrix-vector multiplication, RMS normalization,
  softmax, and vector addition.
- `QwenModel.Forward(...)` runs embedding, RoPE, grouped-query attention,
  SwiGLU feed-forward layers, residual connections, and output logits.
- `KvCache` stores old keys and values used by causal self-attention.
- `SimpleInferenceEngine.AnswerAsync(...)` repeats the forward pass and selects
  the largest logit until Qwen produces its end token.
- `Program.cs` reads command-line values, calls the engine, and prints the
  answer.

This is intentionally a learning example: CPU-only, one request at a time,
deterministic output, and no server, batching, GPU, quantization, SIMD, or other
optimization. The FP16 file expands to ordinary `float[]` arrays in memory, so
it needs several gigabytes of RAM.