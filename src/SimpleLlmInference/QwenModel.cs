namespace SimpleLlmInference;

/// <summary>A direct, single-threaded implementation of the Qwen 2 transformer equations.</summary>
internal sealed class QwenModel
{
    private readonly Tensor _embedding;
    private readonly Tensor _finalNorm;
    private readonly Tensor _output;
    private readonly Layer[] _layers;
    private readonly int _hiddenSize;
    private readonly int _headCount;
    private readonly int _keyValueHeadCount;
    private readonly int _headSize;
    private readonly float _epsilon;
    private readonly float _ropeBase;

    public QwenModel(GgufReader gguf)
    {
        _hiddenSize = gguf.Int("qwen2.embedding_length");
        _headCount = gguf.Int("qwen2.attention.head_count");
        _keyValueHeadCount = gguf.Int("qwen2.attention.head_count_kv");
        _headSize = _hiddenSize / _headCount;
        _epsilon = gguf.Float("qwen2.attention.layer_norm_rms_epsilon");
        _ropeBase = gguf.Float("qwen2.rope.freq_base");

        _embedding = gguf.Tensor("token_embd.weight");
        _finalNorm = gguf.Tensor("output_norm.weight");
        _output = gguf.Tensor("output.weight");

        _layers = new Layer[gguf.Int("qwen2.block_count")];
        for (var layer = 0; layer < _layers.Length; layer++)
        {
            var prefix = $"blk.{layer}.";
            _layers[layer] = new Layer(
                gguf.Tensor(prefix + "attn_norm.weight"),
                gguf.Tensor(prefix + "attn_q.weight"),
                gguf.Tensor(prefix + "attn_q.bias"),
                gguf.Tensor(prefix + "attn_k.weight"),
                gguf.Tensor(prefix + "attn_k.bias"),
                gguf.Tensor(prefix + "attn_v.weight"),
                gguf.Tensor(prefix + "attn_v.bias"),
                gguf.Tensor(prefix + "attn_output.weight"),
                gguf.Tensor(prefix + "ffn_norm.weight"),
                gguf.Tensor(prefix + "ffn_gate.weight"),
                gguf.Tensor(prefix + "ffn_up.weight"),
                gguf.Tensor(prefix + "ffn_down.weight"));
        }
    }

    public KvCache CreateCache(int length) =>
        new(_layers.Length, length, _keyValueHeadCount * _headSize);

    /// <summary>Runs one token through embedding, attention, feed-forward layers, and output logits.</summary>
    public float[] Forward(int tokenId, int position, KvCache cache)
    {
        var hidden = new float[_hiddenSize];
        Array.Copy(_embedding.Data, tokenId * _hiddenSize, hidden, 0, _hiddenSize);

        for (var layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
        {
            var layer = _layers[layerIndex];
            var normalized = TransformerMath.RmsNorm(hidden, layer.AttentionNorm, _epsilon);
            var query = Project(layer.Query, layer.QueryBias, normalized);
            var key = Project(layer.Key, layer.KeyBias, normalized);
            var value = Project(layer.Value, layer.ValueBias, normalized);

            ApplyRope(query, _headCount, position);
            ApplyRope(key, _keyValueHeadCount, position);
            cache.Store(layerIndex, position, key, value);

            var attended = Attention(query, layerIndex, position, cache);
            TransformerMath.AddInPlace(hidden, TransformerMath.MatrixVector(layer.AttentionOutput, attended));

            normalized = TransformerMath.RmsNorm(hidden, layer.FeedForwardNorm, _epsilon);
            var gate = TransformerMath.MatrixVector(layer.FeedForwardGate, normalized);
            var up = TransformerMath.MatrixVector(layer.FeedForwardUp, normalized);

            for (var i = 0; i < gate.Length; i++)
            {
                gate[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i];
            }

            TransformerMath.AddInPlace(
                hidden,
                TransformerMath.MatrixVector(layer.FeedForwardDown, gate));
        }

        var final = TransformerMath.RmsNorm(hidden, _finalNorm, _epsilon);
        return TransformerMath.MatrixVector(_output, final);
    }

    private float[] Attention(float[] query, int layer, int position, KvCache cache)
    {
        var output = new float[_hiddenSize];
        var groupSize = _headCount / _keyValueHeadCount;
        var scale = 1f / MathF.Sqrt(_headSize);

        for (var head = 0; head < _headCount; head++)
        {
            var keyValueHead = head / groupSize;
            var scores = new float[position + 1];

            for (var earlier = 0; earlier <= position; earlier++)
            {
                float score = 0;
                for (var i = 0; i < _headSize; i++)
                {
                    score += query[head * _headSize + i]
                        * cache.Key(layer, earlier, keyValueHead * _headSize + i);
                }
                scores[earlier] = score * scale;
            }

            TransformerMath.SoftmaxInPlace(scores);

            for (var earlier = 0; earlier <= position; earlier++)
            {
                for (var i = 0; i < _headSize; i++)
                {
                    output[head * _headSize + i] += scores[earlier]
                        * cache.Value(layer, earlier, keyValueHead * _headSize + i);
                }
            }
        }

        return output;
    }

    private void ApplyRope(float[] vector, int heads, int position)
    {
        var half = _headSize / 2;
        for (var head = 0; head < heads; head++)
        {
            var start = head * _headSize;
            for (var i = 0; i < half; i++)
            {
                var angle = position / MathF.Pow(_ropeBase, 2f * i / _headSize);
                var cosine = MathF.Cos(angle);
                var sine = MathF.Sin(angle);
                var first = vector[start + i];
                var second = vector[start + half + i];
                vector[start + i] = first * cosine - second * sine;
                vector[start + half + i] = second * cosine + first * sine;
            }
        }
    }

    private static float[] Project(Tensor weights, Tensor bias, float[] input)
    {
        var result = TransformerMath.MatrixVector(weights, input);
        TransformerMath.AddInPlace(result, bias.Data);
        return result;
    }

    private sealed record Layer(
        Tensor AttentionNorm,
        Tensor Query,
        Tensor QueryBias,
        Tensor Key,
        Tensor KeyBias,
        Tensor Value,
        Tensor ValueBias,
        Tensor AttentionOutput,
        Tensor FeedForwardNorm,
        Tensor FeedForwardGate,
        Tensor FeedForwardUp,
        Tensor FeedForwardDown);
}

/// <summary>Stores earlier keys and values so attention can remember previous tokens.</summary>
internal sealed class KvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _width;

    public KvCache(int layers, int length, int width)
    {
        _width = width;
        _keys = Enumerable.Range(0, layers).Select(_ => new float[length * width]).ToArray();
        _values = Enumerable.Range(0, layers).Select(_ => new float[length * width]).ToArray();
    }

    public void Store(int layer, int position, float[] key, float[] value)
    {
        Array.Copy(key, 0, _keys[layer], position * _width, _width);
        Array.Copy(value, 0, _values[layer], position * _width, _width);
    }

    public float Key(int layer, int position, int item) =>
        _keys[layer][position * _width + item];

    public float Value(int layer, int position, int item) =>
        _values[layer][position * _width + item];
}
