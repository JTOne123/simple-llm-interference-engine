namespace SimpleLlmInference;

/// <summary>The small collection of equations used by a Qwen transformer.</summary>
public static class TransformerMath
{
    /// <summary>Multiplies a matrix by a vector: every output is one weighted sum.</summary>
    internal static float[] MatrixVector(Tensor matrix, float[] vector)
    {
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException("Matrix and vector sizes do not match.");
        }

        var result = new float[matrix.Rows];
        for (var row = 0; row < matrix.Rows; row++)
        {
            float sum = 0;
            var start = row * matrix.Columns;
            for (var column = 0; column < matrix.Columns; column++)
            {
                sum += matrix.Data[start + column] * vector[column];
            }
            result[row] = sum;
        }
        return result;
    }

    /// <summary>Keeps a vector's scale stable without changing its direction.</summary>
    internal static float[] RmsNorm(float[] input, Tensor weights, float epsilon)
    {
        float squares = 0;
        for (var i = 0; i < input.Length; i++)
        {
            squares += input[i] * input[i];
        }

        var scale = 1f / MathF.Sqrt(squares / input.Length + epsilon);
        var result = new float[input.Length];
        for (var i = 0; i < input.Length; i++)
        {
            result[i] = input[i] * scale * weights.Data[i];
        }
        return result;
    }

    /// <summary>Turns arbitrary attention scores into positive weights that add up to one.</summary>
    public static void SoftmaxInPlace(float[] values)
    {
        var max = values.Max();
        float total = 0;
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            total += values[i];
        }
        for (var i = 0; i < values.Length; i++)
        {
            values[i] /= total;
        }
    }

    internal static void AddInPlace(float[] left, float[] right)
    {
        for (var i = 0; i < left.Length; i++)
        {
            left[i] += right[i];
        }
    }
}
