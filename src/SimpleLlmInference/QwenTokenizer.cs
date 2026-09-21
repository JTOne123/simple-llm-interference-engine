using System.Text;
using System.Text.RegularExpressions;

namespace SimpleLlmInference;

/// <summary>
/// Qwen's byte-level BPE tokenizer, implemented with ordinary dictionaries and loops.
/// A tokenizer is the translator between human text and the integer IDs understood by the model.
/// </summary>
internal sealed partial class QwenTokenizer
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenIds;
    private readonly Dictionary<string, int> _mergeRanks;
    private readonly char[] _byteEncoder = new char[256];
    private readonly Dictionary<char, byte> _byteDecoder = [];

    /// <summary>The token that means "the generated text is finished".</summary>
    public int EndTokenId { get; }

    /// <summary>The special token that marks the beginning of a chat message.</summary>
    public int ImStartTokenId => _tokenIds["<|im_start|>"];

    /// <summary>The special token that marks the end of a chat message.</summary>
    public int ImEndTokenId => _tokenIds["<|im_end|>"];

    /// <summary>
    /// Loads the vocabulary (token text to token ID), BPE merge priorities, and special IDs
    /// stored inside the GGUF model.
    /// </summary>
    public QwenTokenizer(GgufReader gguf)
    {
        _tokens = gguf.Strings("tokenizer.ggml.tokens");
        _tokenIds = _tokens.Select((token, id) => (token, id))
            .ToDictionary(item => item.token, item => item.id);
        _mergeRanks = gguf.Strings("tokenizer.ggml.merges")
            .Select((merge, rank) => (merge, rank))
            .ToDictionary(item => item.merge, item => item.rank);
        EndTokenId = gguf.Int("tokenizer.ggml.eos_token_id");
        BuildByteMaps();
    }

    /// <summary>
    /// Wraps a question in Qwen's expected chat format and converts it to token IDs.
    ///
    /// Layman version: an instruction model was trained on labelled messages, not bare questions.
    /// We therefore send "system says...", "user says...", then open an empty "assistant says..."
    /// message. The model continues that final message with its answer.
    /// </summary>
    public List<int> EncodeChat(string question)
    {
        var result = new List<int>();
        AddMessage(result, "system", "You are a helpful assistant. Answer briefly and use current English place names.");
        AddMessage(result, "user", question);
        result.Add(ImStartTokenId);
        result.AddRange(EncodeOrdinary("assistant\n"));
        return result;
    }

    /// <summary>
    /// Converts generated token IDs back into readable UTF-8 text.
    /// Tokens are first joined into Qwen's byte-safe character form, then changed back to bytes.
    /// </summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        var encoded = string.Concat(tokenIds.Select(id => _tokens[id]));
        using var bytes = new MemoryStream();

        foreach (var character in encoded)
        {
            if (_byteDecoder.TryGetValue(character, out var value))
            {
                bytes.WriteByte(value);
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>
    /// Adds one labelled chat message: start marker, role, text, end marker, and newline.
    /// Keeping this exact format matters because it matches the examples Qwen learned from.
    /// </summary>
    private void AddMessage(List<int> result, string role, string text)
    {
        result.Add(ImStartTokenId);
        result.AddRange(EncodeOrdinary($"{role}\n{text}"));
        result.Add(ImEndTokenId);
        result.AddRange(EncodeOrdinary("\n"));
    }

    /// <summary>
    /// Converts normal text to token IDs in two stages:
    /// first split text into word-like chunks, then apply BPE merges inside each chunk.
    /// </summary>
    private IEnumerable<int> EncodeOrdinary(string text)
    {
        foreach (Match match in TokenPattern().Matches(text))
        {
            var encoded = string.Concat(Encoding.UTF8.GetBytes(match.Value).Select(value => _byteEncoder[value]));
            foreach (var piece in ApplyBpe(encoded))
            {
                yield return _tokenIds[piece];
            }
        }
    }

    /// <summary>
    /// Repeatedly joins the most preferred neighboring pieces according to Qwen's merge table.
    ///
    /// Layman example: a word starts as individual characters. If "K y" has a better learned
    /// rank than other pairs, it becomes "Ky"; later "Ky i" may become "Kyi". The process stops
    /// when no known neighboring pair remains. The final pieces are vocabulary tokens.
    /// </summary>
    private List<string> ApplyBpe(string word)
    {
        var pieces = word.Select(character => character.ToString()).ToList();

        while (pieces.Count > 1)
        {
            var bestRank = int.MaxValue;
            string? bestPair = null;

            for (var i = 0; i < pieces.Count - 1; i++)
            {
                var pair = $"{pieces[i]} {pieces[i + 1]}";
                if (_mergeRanks.TryGetValue(pair, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestPair = pair;
                }
            }

            if (bestPair is null)
            {
                break;
            }

            var separator = bestPair.IndexOf(' ');
            var left = bestPair[..separator];
            var right = bestPair[(separator + 1)..];
            var merged = new List<string>();

            for (var i = 0; i < pieces.Count;)
            {
                if (i < pieces.Count - 1 && pieces[i] == left && pieces[i + 1] == right)
                {
                    merged.Add(left + right);
                    i += 2;
                }
                else
                {
                    merged.Add(pieces[i++]);
                }
            }
            pieces = merged;
        }

        return pieces;
    }

    /// <summary>
    /// Builds a reversible mapping for all 256 possible byte values.
    ///
    /// BPE works with text-like strings, but arbitrary UTF-8 bytes are not all printable.
    /// This mapping gives every byte a safe character representation without losing information.
    /// </summary>
    private void BuildByteMaps()
    {
        var visible = Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
            .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
            .ToHashSet();
        var extra = 0;

        for (var value = 0; value < 256; value++)
        {
            var character = (char)(visible.Contains(value) ? value : 256 + extra++);
            _byteEncoder[value] = character;
            _byteDecoder[character] = (byte)value;
        }
    }

    /// <summary>
    /// Defines Qwen's first-pass text chunks: contractions, words, numbers, punctuation,
    /// and whitespace. BPE merging is performed separately inside each matched chunk.
    /// </summary>
    [GeneratedRegex(@"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex TokenPattern();
}
