using SimpleLlmInference;

const string defaultModelPath =
    @"D:\LLMModels\qwen2.5-0.5b-instruct-fp16.gguf";

var modelPath = args.FirstOrDefault() ?? defaultModelPath;
var question = args.Skip(1).FirstOrDefault() ?? "What is the capital of Ukraine?";

Console.WriteLine($"Question: {question}");
Console.Write("Answer: ");

using var engine = new SimpleInferenceEngine(modelPath);
var answer = await engine.AnswerAsync(question);
Console.WriteLine(answer);
