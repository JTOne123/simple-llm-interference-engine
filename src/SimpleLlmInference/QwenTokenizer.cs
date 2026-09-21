using System.Text;
using System.Text.RegularExpressions;

namespace SimpleLlmInference;

/// <summary>Qwen's byte-level BPE tokenizer, implemented with ordinary dictionaries and loops.</summary>
internal sealed partial class QwenTokenizer
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenIds;
    private readonly Dictionary<string, int> _mergeRanks;
    private readonly char[] _byteEncoder = new char[256];
    private readonly Dictionary<char, byte> _byteDecoder = [];

    public int EndTokenId { get; }
    public int ImStartTokenId => _tokenIds["<|im_start|>"];
    public int ImEndTokenId => _tokenIds["<|im_end|>"];

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

    /// <summary>Wraps a question in Qwen's simple system/user/assistant chat format.</summary>
    public List<int> EncodeChat(string question)
    {
        var result = new List<int>();
        AddMessage(result, "system", "You are a helpful assistant. Answer briefly and use current English place names.");
        AddMessage(result, "user", question);
        result.Add(ImStartTokenId);
        result.AddRange(EncodeOrdinary("assistant\n"));
        return result;
    }

    /// <summary>Converts generated token IDs back into readable UTF-8 text.</summary>
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

    private void AddMessage(List<int> result, string role, string text)
    {
        result.Add(ImStartTokenId);
        result.AddRange(EncodeOrdinary($"{role}\n{text}"));
        result.Add(ImEndTokenId);
        result.AddRange(EncodeOrdinary("\n"));
    }

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

    [GeneratedRegex(@"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex TokenPattern();
}
