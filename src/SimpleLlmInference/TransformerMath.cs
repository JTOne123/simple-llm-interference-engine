namespace SimpleLlmInference;

/// <summary>
/// The small collection of equations used by a Qwen transformer.
/// These methods deliberately use ordinary loops so every calculation is visible.
/// </summary>
public static class TransformerMath
{
    /// <summary>
    /// Multiplies a matrix by a vector.
    ///
    /// Layman version: every matrix row is a learned recipe. Each input number is multiplied
    /// by the recipe's matching weight, then all products are added. One recipe produces one
    /// output number. Neural networks use this operation to transform one meaning-vector into
    /// another, such as turning a token representation into a query vector.
    ///
    /// Formula for output row r: result[r] = sum(matrix[r,c] * vector[c]).
    /// </summary>
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

            // Calculate the dot product between this matrix row and the input vector.
            for (var column = 0; column < matrix.Columns; column++)
            {
                sum += matrix.Data[start + column] * vector[column];
            }
            result[row] = sum;
        }
        return result;
    }

    /// <summary>
    /// Keeps a vector's size under control without erasing the information in its pattern.
    ///
    /// Layman version: numbers can grow after passing through many layers. RMS normalization
    /// measures their typical size and shrinks or enlarges the whole vector to a predictable
    /// scale. The learned weights then let the model adjust each position independently.
    ///
    /// Formula: output[i] = input[i] / sqrt(mean(input²) + epsilon) * weight[i].
    /// Epsilon is a tiny safety value that prevents division by zero.
    /// </summary>
    internal static float[] RmsNorm(float[] input, Tensor weights, float epsilon)
    {
        float squares = 0;
        for (var i = 0; i < input.Length; i++)
        {
            squares += input[i] * input[i];
        }

        // This is 1 / root-mean-square, calculated once and reused for every item.
        var scale = 1f / MathF.Sqrt(squares / input.Length + epsilon);
        var result = new float[input.Length];
        for (var i = 0; i < input.Length; i++)
        {
            result[i] = input[i] * scale * weights.Data[i];
        }
        return result;
    }

    /// <summary>
    /// Turns arbitrary scores into positive shares that add up to one.
    ///
    /// Layman version: attention first produces scores such as [-2, 1, 4]. Softmax converts
    /// them into percentages such as [0.2%, 4.7%, 95.1%]. The largest score gets the most
    /// attention, but smaller scores can still contribute.
    ///
    /// Subtracting the largest score before exponentiation does not change the percentages;
    /// it only prevents very large numbers from overflowing.
    /// </summary>
    public static void SoftmaxInPlace(float[] values)
    {
        var max = values.Max();
        float total = 0;

        // Exponentiation makes every value positive and strongly favors larger scores.
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            total += values[i];
        }

        // Dividing by the total turns the values into fractions that sum to exactly one.
        for (var i = 0; i < values.Length; i++)
        {
            values[i] /= total;
        }
    }

    /// <summary>
    /// Adds one vector into another item by item.
    ///
    /// Transformers call this a residual connection. It preserves the information that was
    /// already present while adding the new information produced by attention or feed-forward
    /// processing. In plain terms: keep the old thought and add the newly learned correction.
    /// </summary>
    internal static void AddInPlace(float[] left, float[] right)
    {
        for (var i = 0; i < left.Length; i++)
        {
            left[i] += right[i];
        }
    }
}
