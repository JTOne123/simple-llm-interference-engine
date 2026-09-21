using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SimpleLlmInference.Tests;

[TestClass]
public sealed class SimpleInferenceEngineTests
{
    private const string ModelPath =
        @"D:\LLMModels\qwen2.5-0.5b-instruct-fp16.gguf";

    [TestMethod]
    public void SoftmaxProducesOrderedProbabilitiesThatSumToOne()
    {
        float[] values = [1, 2, 3];

        TransformerMath.SoftmaxInPlace(values);

        Assert.AreEqual(1f, values.Sum(), 0.0001f);
        Assert.IsTrue(values[2] > values[1] && values[1] > values[0]);
    }

    [TestMethod]
    public void MissingModelIsRejected()
    {
        Assert.ThrowsExactly<FileNotFoundException>(
            () => new SimpleInferenceEngine("missing-model.gguf"));
    }

    [TestMethod]
    [TestCategory("Integration")]
    public async Task AnswersCapitalOfUkraine()
    {
        using var engine = new SimpleInferenceEngine(ModelPath);

        var answer = await engine.AnswerAsync(
            "Answer with one word only. What is the capital of Ukraine?",
            maxTokens: 8);

        Assert.IsTrue(
            answer.Contains("Kyiv", StringComparison.OrdinalIgnoreCase),
            $"Expected Kyiv, got: {answer}");
    }
}
