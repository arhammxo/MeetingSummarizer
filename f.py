def compute_context_recall(gt, contexts):
    claims = extract_claims(gt)
    covered = 0
    for claim in claims:
        for ctx in contexts:
            if "not supported" not in call_llm(
                f"Is this claim supported by the context?\n"
                f"Context: {ctx}\nClaim: {claim}\n"
                f"Answer only: supported / not supported"
            ):
                covered += 1
                break
    return covered / len(claims) if claims else 0

def compute_ faithfulness(hyp, contexts):
    """
    Check how much of the generated summary (hyp) is supported by contexts.
    """
    claims = extract_claims(hyp)
    covered = 0
    for claim in claims:
        if any(
            "not supported" not in call_llm(
                f"Is this claim supported by the context?\nContext: {ctx}\nClaim: {claim}\nAnswer only: supported / not supported"
            )
            for ctx in contexts
        ):
            covered += 1
    return covered / len(claims) if claims else 0

def compute_response_faithfulness(hyp, contexts):
    claims = extract_claims(hyp)
    covered = 0
    for claim in claims:
        if any(
            "not supported" not in call_llm(
                f"Is this claim supported by the context?\nContext: {ctx}\nClaim: {claim}\nAnswer only: supported / not supported"
            )
            for ctx in contexts
        ):
            covered += 1
    return covered / len(claims) if claims else 0

def compute_response_relevance(hyp, gt):
    claims = extract_claims(hyp)
    covered = 0
    for claim in claims:
        if any(
            "not supported" not in call_llm(
                f"Is this claim relevant to the ground truth?\nContext: {gt}\nClaim: {claim}\nAnswer only: relevant / not relevant"
            )
            for ctx in contexts
        ):
            covered += 1
    return covered / len(claims) if claims else 0   

import json
def load_dataset(path):
    return json.load(open(path))["results"]

    def compute_metrics(dataset_path, model_name, output_path="metrics.json"):
    dataset = load_dataset(dataset_path)
    results = []
    for entry in dataset:
        contexts = entry["contexts"]
        gt = entry["ground_truth"]

        # Generate summaries from the model you want to evaluate
        abstractive_summary = generate_abstractive(contexts, gt, model_name)
        extractive_summary = generate_extractive(contexts, gt, model_name)
        cot_summary = generate_cot(contexts, gt, model_name)
        qa_summary = generate_qa(contexts, gt, model_name)

        # Compute metrics for each
        abstractive_metrics = compute_all_metrics(abstractive_summary, contexts, gt)
        extractive_metrics = compute_all_metrics(extractive_summary, contexts, gt)
        cot_metrics = compute_all_metrics(cot_summary, contexts, gt)
        qa_metrics = compute_all_metrics(qa_summary, contexts, gt)

        results.append({
            "contexts": contexts,
            "ground_truth": gt,
            "abstractive_summary": abstractive_summary,
            "extractive_summary": extractive_summary,
            "cot_summary": cot_summary,
            "qa_summary": qa_summary,
            "abstractive_metrics": abstractive_metrics,
            "extractive_metrics": extractive_metrics,
            "cot_metrics": cot_metrics,
            "qa_metrics": qa_metrics,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# -------------------------------
# Run evaluation
# -------------------------------
if __name__ == "__main__":
    compute_metrics(
        dataset_path="human_eval.json",
        model_name="t5-base",
        output_path="eval_results.json"
    )