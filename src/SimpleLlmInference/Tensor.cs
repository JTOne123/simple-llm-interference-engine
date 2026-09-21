namespace SimpleLlmInference;

/// <summary>A plain multidimensional array. Dimensions describe the original GGUF tensor.</summary>
internal sealed record Tensor(float[] Data, int[] Dimensions)
{
    public int Columns => Dimensions[0];
    public int Rows => Dimensions.Length > 1 ? Dimensions[1] : 1;
}
