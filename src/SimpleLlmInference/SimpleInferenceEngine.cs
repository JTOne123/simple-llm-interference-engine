namespace SimpleLlmInference;

/// <summary>
/// A small, pure C# Qwen 2.5 inference engine with no runtime packages.
/// It coordinates file loading, tokenization, transformer execution, and token generation.
/// </summary>
public sealed class SimpleInferenceEngine : IDisposable
{
    private readonly GgufReader _gguf;
    private readonly QwenTokenizer _tokenizer;
    private readonly QwenModel _model;

    /// <summary>
    /// Opens the GGUF file, loads Qwen's tokenizer, and loads all transformer weights.
    ///
    /// Layman version: this prepares the model's dictionary and billions of learned numbers
    /// before any question is asked. The FP16 weights are expanded to normal C# floats, so this
    /// simple version uses more memory than optimized engines.
    /// </summary>
    public SimpleInferenceEngine(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("The GGUF model file was not found.", modelPath);
        }

        _gguf = new GgufReader(modelPath);
        _tokenizer = new QwenTokenizer(_gguf);
        _model = new QwenModel(_gguf);
    }

    /// <summary>
    /// Answers a question by repeatedly predicting one token at a time.
    ///
    /// First, every prompt token is passed through the model to fill its KV cache (the prefill
    /// phase). Then the largest output score is chosen, that token is passed back through the
    /// model, and the cycle repeats (the decode phase). Generation ends at a stop token or at
    /// <paramref name="maxTokens"/>.
    /// </summary>
    public Task<string> AnswerAsync(
        string question,
        int maxTokens = 32,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(question))
        {
            throw new ArgumentException("A question is required.", nameof(question));
        }

        var prompt = _tokenizer.EncodeChat(question);
        var cache = _model.CreateCache(prompt.Count + maxTokens);
        float[]? logits = null;
        var position = 0;

        // Prefill: let the model read the complete question and remember it in the KV cache.
        foreach (var token in prompt)
        {
            cancellationToken.ThrowIfCancellationRequested();
            logits = _model.Forward(token, position++, cache);
        }

        var answerTokens = new List<int>();

        // Decode: select one answer token, feed it back, and ask for the next token.
        for (var i = 0; i < maxTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var nextToken = ArgMax(logits!);
            if (nextToken == _tokenizer.EndTokenId || nextToken == _tokenizer.ImEndTokenId)
            {
                break;
            }

            answerTokens.Add(nextToken);
            logits = _model.Forward(nextToken, position++, cache);
        }

        return Task.FromResult(_tokenizer.Decode(answerTokens).Trim());
    }

    /// <summary>
    /// Finds the vocabulary item with the largest raw model score (logit).
    ///
    /// This is greedy decoding: always take the model's first choice. It is deterministic and
    /// easy to understand, unlike temperature or random sampling.
    /// </summary>
    private static int ArgMax(float[] logits)
    {
        var best = 0;
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Closes the GGUF file. The large managed arrays become reclaimable by .NET's garbage
    /// collector after the engine is no longer referenced.
    /// </summary>
    public void Dispose() => _gguf.Dispose();
}
