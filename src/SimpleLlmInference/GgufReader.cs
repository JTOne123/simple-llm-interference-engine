using System.Text;

namespace SimpleLlmInference;

/// <summary>
/// Reads metadata and FP16/FP32 tensors from one GGUF file.
/// GGUF is a container: its header describes the model and its large data section stores weights.
/// </summary>
internal sealed class GgufReader : IDisposable
{
    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private readonly Dictionary<string, object> _metadata = [];
    private readonly Dictionary<string, TensorInfo> _tensors = [];
    private readonly Dictionary<string, Tensor> _loaded = [];
    private long _tensorDataStart;

    /// <summary>
    /// Opens the model file and reads only its table of contents.
    /// The large tensors are loaded later, when <see cref="Tensor"/> is called.
    /// </summary>
    public GgufReader(string path)
    {
        _stream = File.OpenRead(path);
        _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);
        ReadHeader();
    }

    /// <summary>Reads an integer model setting, such as the number of transformer layers.</summary>
    public int Int(string key) => Convert.ToInt32(_metadata[key]);

    /// <summary>Reads a decimal model setting, such as the small RMSNorm epsilon value.</summary>
    public float Float(string key) => Convert.ToSingle(_metadata[key]);

    /// <summary>Reads a list of strings, such as tokenizer vocabulary pieces or merge rules.</summary>
    public string[] Strings(string key) => (string[])_metadata[key];

    /// <summary>
    /// Loads one named tensor and converts its numbers to ordinary C# <see cref="float"/> values.
    ///
    /// Layman version: the GGUF header tells us where each giant number table starts. We jump
    /// to that location, read every number, and remember the result so later requests do not
    /// read it again. FP16 uses two bytes per number; we expand it to the easier four-byte float.
    /// </summary>
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

    /// <summary>
    /// Reads the GGUF table of contents: file version, general settings, tensor names, shapes,
    /// number formats, and byte offsets. It does not yet read the large tensor values.
    /// </summary>
    private void ReadHeader()
    {
        // "GGUF" written as four bytes is the file signature that identifies the format.
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

        // Metadata contains small descriptive values: architecture, layer count, tokenizer, etc.
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ReadString();
            var type = _reader.ReadUInt32();
            _metadata[key] = ReadValue(type);
        }

        // Tensor descriptors tell us the shape and file location of every learned weight table.
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

    /// <summary>
    /// Reads one metadata value according to the numeric GGUF type code stored before it.
    /// This is similar to reading a JSON value after learning whether it is a number or string.
    /// </summary>
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

    /// <summary>
    /// Reads a GGUF metadata array. Token lists are string arrays; other rare arrays are kept
    /// as general object arrays because this small engine does not need stronger types for them.
    /// </summary>
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

    /// <summary>
    /// Reads a GGUF string, which is stored as an eight-byte length followed by UTF-8 bytes.
    /// </summary>
    private string ReadString()
    {
        var length = checked((int)_reader.ReadUInt64());
        return Encoding.UTF8.GetString(_reader.ReadBytes(length));
    }

    /// <summary>
    /// Moves a byte position to the next required boundary.
    /// GGUF pads sections so tensors begin at clean, predictable addresses.
    /// </summary>
    private static long Align(long value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    /// <summary>Closes the binary reader and the underlying model file.</summary>
    public void Dispose()
    {
        _reader.Dispose();
        _stream.Dispose();
    }

    private sealed record TensorInfo(int[] Dimensions, uint Type, long Offset);
}
