namespace SimpleLlmInference;

/// <summary>
/// A direct, single-threaded implementation of the Qwen 2 transformer equations.
///
/// The model moves a token through many identical layers. Each layer first uses attention to
/// gather useful facts from earlier tokens, then uses a feed-forward network to process that
/// gathered information. Learned weight tensors decide what each transformation means.
/// </summary>
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

    /// <summary>
    /// Reads Qwen's architecture settings and all learned weight tensors from GGUF.
    ///
    /// Layman version: settings describe the machine's shape (layers, heads, vector sizes);
    /// tensors are the learned knowledge that fills that shape.
    /// </summary>
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
            // Every transformer block has the same kinds of weights, but learned values differ.
            var prefix = $"blk.{layer}.";
            _layers[layer] = new Layer(
                // Stabilizes values so attention scores do not become too large or too small.
                AttentionNorm: gguf.Tensor(prefix + "attn_norm.weight"),
                // Creates "What am I looking for?" so this token can find relevant earlier tokens.
                Query: gguf.Tensor(prefix + "attn_q.weight"),
                // Fine-tunes each query value so searching is not limited to matrix mixing alone.
                QueryBias: gguf.Tensor(prefix + "attn_q.bias"),
                // Creates "What do I contain?" labels that queries can compare against.
                Key: gguf.Tensor(prefix + "attn_k.weight"),
                // Fine-tunes each key value so tokens can advertise their information accurately.
                KeyBias: gguf.Tensor(prefix + "attn_k.bias"),
                // Creates the content to retrieve when attention decides this token is relevant.
                Value: gguf.Tensor(prefix + "attn_v.weight"),
                // Fine-tunes retrieved content instead of relying only on matrix multiplication.
                ValueBias: gguf.Tensor(prefix + "attn_v.bias"),
                // Merges attention-head results into hidden size so they can rejoin the main state.
                AttentionOutput: gguf.Tensor(prefix + "attn_output.weight"),
                // Stabilizes values so the feed-forward network receives a predictable scale.
                FeedForwardNorm: gguf.Tensor(prefix + "ffn_norm.weight"),
                // Opens useful features and suppresses irrelevant ones before they affect the state.
                FeedForwardGate: gguf.Tensor(prefix + "ffn_gate.weight"),
                // Expands the vector to give the model room to recognize more complex features.
                FeedForwardUp: gguf.Tensor(prefix + "ffn_up.weight"),
                // Returns features to hidden size so they can be added through the residual path.
                FeedForwardDown: gguf.Tensor(prefix + "ffn_down.weight"));
        }
    }

    /// <summary>
    /// Allocates memory for keys and values from every token and every layer.
    /// Without this cache, generating each new token would recalculate the entire prompt.
    /// </summary>
    public KvCache CreateCache(int length) =>
        new(_layers.Length, length, _keyValueHeadCount * _headSize);

    /// <summary>
    /// Runs one token through the complete transformer and returns one score per vocabulary token.
    ///
    /// The returned scores are logits. They are not percentages; they only express relative
    /// preference. The inference engine selects the vocabulary token with the largest logit.
    /// </summary>
    public float[] Forward(int tokenId, int position, KvCache cache)
    {
        // Embedding lookup: copy the learned meaning-vector belonging to this token ID.
        var hidden = new float[_hiddenSize];
        Array.Copy(_embedding.Data, tokenId * _hiddenSize, hidden, 0, _hiddenSize);

        for (var layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
        {
            var layer = _layers[layerIndex];

            // Normalize before attention so the following calculations receive stable values.
            var normalized = TransformerMath.RmsNorm(hidden, layer.AttentionNorm, _epsilon);

            // Q asks "what am I looking for?", K says "what do I contain?", and V holds
            // the information to copy when a query considers a key relevant.
            var query = Project(layer.Query, layer.QueryBias, normalized);
            var key = Project(layer.Key, layer.KeyBias, normalized);
            var value = Project(layer.Value, layer.ValueBias, normalized);

            // RoPE writes token position into Q and K, allowing attention to understand order.
            ApplyRope(query, _headCount, position);
            ApplyRope(key, _keyValueHeadCount, position);

            // Save this token's K and V so this and future tokens can attend to them.
            cache.Store(layerIndex, position, key, value);

            // Attention gathers a weighted mixture of information from current and earlier tokens.
            var attended = Attention(query, layerIndex, position, cache);

            // Project the gathered information back to hidden size and add it to the old state.
            TransformerMath.AddInPlace(hidden, TransformerMath.MatrixVector(layer.AttentionOutput, attended));

            // The second half of the layer is a small neural network applied to this token alone.
            normalized = TransformerMath.RmsNorm(hidden, layer.FeedForwardNorm, _epsilon);
            var gate = TransformerMath.MatrixVector(layer.FeedForwardGate, normalized);
            var up = TransformerMath.MatrixVector(layer.FeedForwardUp, normalized);

            // SwiGLU: SiLU(gate) * up. The gate decides which learned features may pass through.
            for (var i = 0; i < gate.Length; i++)
            {
                gate[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i];
            }

            // Reduce the wider feed-forward vector back to hidden size and keep a residual path.
            TransformerMath.AddInPlace(
                hidden,
                TransformerMath.MatrixVector(layer.FeedForwardDown, gate));
        }

        // Convert the final hidden meaning into one raw next-token score for every vocabulary item.
        var final = TransformerMath.RmsNorm(hidden, _finalNorm, _epsilon);
        return TransformerMath.MatrixVector(_output, final);
    }

    /// <summary>
    /// Finds which earlier tokens matter to the current token and combines their value vectors.
    ///
    /// For each attention head:
    /// 1. Dot the current query with every cached key to measure similarity.
    /// 2. Divide by sqrt(head size) so larger vectors do not create excessively large scores.
    /// 3. Apply softmax to turn scores into attention percentages.
    /// 4. Build a weighted average of the matching cached values.
    ///
    /// Qwen uses grouped-query attention: several query heads share one key/value head. This
    /// reduces cache size while preserving multiple ways to ask questions of the context.
    /// </summary>
    private float[] Attention(float[] query, int layer, int position, KvCache cache)
    {
        var output = new float[_hiddenSize];
        var groupSize = _headCount / _keyValueHeadCount;

        // Dot products grow with vector length; this standard scale keeps softmax well behaved.
        var scale = 1f / MathF.Sqrt(_headSize);

        for (var head = 0; head < _headCount; head++)
        {
            // Multiple query heads intentionally reuse the same smaller K/V head.
            var keyValueHead = head / groupSize;
            var scores = new float[position + 1];

            for (var earlier = 0; earlier <= position; earlier++)
            {
                float score = 0;

                // A dot product is high when query and key point in similar directions.
                for (var i = 0; i < _headSize; i++)
                {
                    score += query[head * _headSize + i]
                        * cache.Key(layer, earlier, keyValueHead * _headSize + i);
                }
                scores[earlier] = score * scale;
            }

            TransformerMath.SoftmaxInPlace(scores);

            // Mix earlier value vectors according to the attention percentages.
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

    /// <summary>
    /// Encodes token position by rotating pairs of numbers in every query or key head.
    ///
    /// Layman version: the same word at positions 1 and 20 begins with the same embedding.
    /// RoPE turns its Q/K coordinates by position-dependent angles, like moving clock hands.
    /// Attention can then distinguish order and relative distance without adding a separate
    /// position vector. Different coordinate pairs rotate at different speeds.
    ///
    /// Rotation formula:
    /// newFirst  = first * cos(angle) - second * sin(angle)
    /// newSecond = second * cos(angle) + first  * sin(angle)
    /// </summary>
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

    /// <summary>
    /// Applies a learned linear transformation and then adds its learned bias.
    ///
    /// Layman version: the matrix mixes the input numbers according to learned recipes; the bias
    /// then nudges each result up or down. Qwen uses this helper to create Q, K, and V vectors.
    /// </summary>
    private static float[] Project(Tensor weights, Tensor bias, float[] input)
    {
        var result = TransformerMath.MatrixVector(weights, input);
        TransformerMath.AddInPlace(result, bias.Data);
        return result;
    }

    /// <summary>
    /// Holds all learned tensors belonging to one transformer block.
    /// This record has no behavior; it simply gives descriptive names to the weight tables.
    /// </summary>
    /// <param name="AttentionNorm">
    /// Stabilizes hidden values so attention scores remain numerically well behaved.
    /// </param>
    /// <param name="Query">
    /// Describes what this token needs so it can find relevant earlier tokens.
    /// </param>
    /// <param name="QueryBias">
    /// Fine-tunes query values beyond what matrix multiplication can express alone.
    /// </param>
    /// <param name="Key">
    /// Describes what each token contains so queries have searchable labels to compare.
    /// </param>
    /// <param name="KeyBias">
    /// Fine-tunes key values so tokens can advertise their information accurately.
    /// </param>
    /// <param name="Value">
    /// Creates the content attention retrieves after deciding that a token is relevant.
    /// </param>
    /// <param name="ValueBias">
    /// Fine-tunes the retrieved content beyond the value matrix calculation.
    /// </param>
    /// <param name="AttentionOutput">
    /// Merges attention-head results into hidden size so they can rejoin the main state.
    /// </param>
    /// <param name="FeedForwardNorm">
    /// Stabilizes values so the feed-forward network receives a predictable input scale.
    /// </param>
    /// <param name="FeedForwardGate">
    /// Allows useful generated features through while suppressing irrelevant features.
    /// </param>
    /// <param name="FeedForwardUp">
    /// Creates a wider feature space so the layer can recognize more complex patterns.
    /// </param>
    /// <param name="FeedForwardDown">
    /// Returns features to hidden size so they can be added through the residual connection.
    /// </param>
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

/// <summary>
/// Stores earlier keys and values so attention can remember previous tokens.
///
/// A key is a searchable label for a token; a value is the information retrieved when that label
/// matches a query. The cache is separated by layer because every layer creates different K/V data.
/// </summary>
internal sealed class KvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _width;

    /// <summary>
    /// Allocates one flat key array and one flat value array per transformer layer.
    /// Each array is logically shaped as [token position, key/value vector item].
    /// </summary>
    public KvCache(int layers, int length, int width)
    {
        _width = width;
        _keys = Enumerable.Range(0, layers).Select(_ => new float[length * width]).ToArray();
        _values = Enumerable.Range(0, layers).Select(_ => new float[length * width]).ToArray();
    }

    /// <summary>
    /// Copies the current token's complete key and value vectors into their layer and position.
    /// Future tokens will read these saved values instead of recomputing earlier tokens.
    /// </summary>
    public void Store(int layer, int position, float[] key, float[] value)
    {
        Array.Copy(key, 0, _keys[layer], position * _width, _width);
        Array.Copy(value, 0, _values[layer], position * _width, _width);
    }

    /// <summary>Reads one number from a previously stored key vector.</summary>
    public float Key(int layer, int position, int item) =>
        _keys[layer][position * _width + item];

    /// <summary>Reads one number from a previously stored value vector.</summary>
    public float Value(int layer, int position, int item) =>
        _values[layer][position * _width + item];
}
