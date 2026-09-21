using System.Text;

namespace SimpleLlmInference;

/// <summary>Reads metadata and FP16/FP32 tensors from one GGUF file.</summary>
internal sealed class GgufReader : IDisposable
{
    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private readonly Dictionary<string, object> _metadata = [];
    private readonly Dictionary<string, TensorInfo> _tensors = [];
    private readonly Dictionary<string, Tensor> _loaded = [];
    private long _tensorDataStart;

    public GgufReader(string path)
    {
        _stream = File.OpenRead(path);
        _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);
        ReadHeader();
    }

    public int Int(string key) => Convert.ToInt32(_metadata[key]);
    public float Float(string key) => Convert.ToSingle(_metadata[key]);
    public string[] Strings(string key) => (string[])_metadata[key];

    /// <summary>Loads a tensor and converts simple FP16 values to ordinary C# floats.</summary>
    public Tensor Tensor(string name)
    {
        if (_loaded.TryGetValue(name, out var loaded))
        {
            return loaded;
        }

        var info = _tensors[name];
        _stream.Position = _tensorDataStart + info.Offset;
        var count = info.Dimensions.Aggregate(1L, (total, size) => total * size);
        var data = new float[checked((int)count)];

        if (info.Type == 0)
        {
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = _reader.ReadSingle();
            }
        }
        else if (info.Type == 1)
        {
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = (float)BitConverter.UInt16BitsToHalf(_reader.ReadUInt16());
            }
        }
        else
        {
            throw new NotSupportedException(
                $"Tensor '{name}' uses GGML type {info.Type}. This educational engine supports only FP32 and FP16.");
        }

        return _loaded[name] = new Tensor(data, info.Dimensions);
    }

    private void ReadHeader()
    {
        if (_reader.ReadUInt32() != 0x46554747)
        {
            throw new InvalidDataException("The file is not GGUF.");
        }

        var version = _reader.ReadUInt32();
        if (version is < 2 or > 3)
        {
            throw new NotSupportedException($"GGUF version {version} is not supported.");
        }

        var tensorCount = _reader.ReadUInt64();
        var metadataCount = _reader.ReadUInt64();

        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ReadString();
            var type = _reader.ReadUInt32();
            _metadata[key] = ReadValue(type);
        }

        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadString();
            var dimensionCount = _reader.ReadUInt32();
            var dimensions = new int[dimensionCount];

            for (var dimension = 0; dimension < dimensions.Length; dimension++)
            {
                dimensions[dimension] = checked((int)_reader.ReadUInt64());
            }

            var type = _reader.ReadUInt32();
            var offset = checked((long)_reader.ReadUInt64());
            _tensors[name] = new TensorInfo(dimensions, type, offset);
        }

        var alignment = _metadata.TryGetValue("general.alignment", out var value)
            ? Convert.ToInt32(value)
            : 32;
        _tensorDataStart = Align(_stream.Position, alignment);
    }

    private object ReadValue(uint type) => type switch
    {
        0 => _reader.ReadByte(),
        1 => _reader.ReadSByte(),
        2 => _reader.ReadUInt16(),
        3 => _reader.ReadInt16(),
        4 => _reader.ReadUInt32(),
        5 => _reader.ReadInt32(),
        6 => _reader.ReadSingle(),
        7 => _reader.ReadByte() != 0,
        8 => ReadString(),
        9 => ReadArray(),
        10 => _reader.ReadUInt64(),
        11 => _reader.ReadInt64(),
        12 => _reader.ReadDouble(),
        _ => throw new NotSupportedException($"Unknown GGUF metadata type {type}.")
    };

    private object ReadArray()
    {
        var elementType = _reader.ReadUInt32();
        var count = checked((int)_reader.ReadUInt64());

        if (elementType == 8)
        {
            var strings = new string[count];
            for (var i = 0; i < count; i++)
            {
                strings[i] = ReadString();
            }
            return strings;
        }

        var values = new object[count];
        for (var i = 0; i < count; i++)
        {
            values[i] = ReadValue(elementType);
        }
        return values;
    }

    private string ReadString()
    {
        var length = checked((int)_reader.ReadUInt64());
        return Encoding.UTF8.GetString(_reader.ReadBytes(length));
    }

    private static long Align(long value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    public void Dispose()
    {
        _reader.Dispose();
        _stream.Dispose();
    }

    private sealed record TensorInfo(int[] Dimensions, uint Type, long Offset);
}
