namespace SimpleLlmInference;

/// <summary>A small, pure C# Qwen 2.5 inference engine with no external packages.</summary>
public sealed class SimpleInferenceEngine : IDisposable
{
    private readonly GgufReader _gguf;
    private readonly QwenTokenizer _tokenizer;
    private readonly QwenModel _model;

    /// <summary>Reads the GGUF metadata, tokenizer, and FP16 model weights into memory.</summary>
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

    /// <summary>Predicts one token at a time and always chooses the highest logit.</summary>
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

        foreach (var token in prompt)
        {
            cancellationToken.ThrowIfCancellationRequested();
            logits = _model.Forward(token, position++, cache);
        }

        var answerTokens = new List<int>();
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

    /// <summary>Finds the vocabulary item with the largest raw model score.</summary>
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

    /// <summary>Closes the GGUF file. All model memory is ordinary managed C# memory.</summary>
    public void Dispose() => _gguf.Dispose();
}
