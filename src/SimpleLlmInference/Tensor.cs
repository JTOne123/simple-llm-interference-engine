namespace SimpleLlmInference;

/// <summary>
/// A tensor is just a box of numbers with a shape.
/// We store all numbers in one flat array because that is the simplest representation in C#.
/// </summary>
internal sealed record Tensor(float[] Data, int[] Dimensions)
{
    /// <summary>
    /// Number of input values used by one matrix row.
    /// GGUF stores the fastest-changing dimension first, so dimension zero is the column count.
    /// </summary>
    public int Columns => Dimensions[0];

    /// <summary>
    /// Number of results produced by the matrix.
    /// A one-dimensional tensor is treated as a single row, which also works for bias vectors.
    /// </summary>
    public int Rows => Dimensions.Length > 1 ? Dimensions[1] : 1;
}
