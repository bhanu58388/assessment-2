import argparse

from models.qa_t5 import T5_Question_Answering


def main():
    parser = argparse.ArgumentParser(description="Run a quick local TELLER yes/no demo.")
    parser.add_argument("claim", help="Claim text to ask the model about")
    parser.add_argument("--evidence", default=None, help="Optional context to include with the claim")
    parser.add_argument("--model", default="flan-t5-small", choices=["flan-t5-small", "flan-t5-base"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    model = T5_Question_Answering(model_name=args.model, device=args.device)
    score = model.answer_direct_sampling(
        info=args.evidence,
        gq=f"Message: {args.claim}\nIs this message false?",
        do_sample=False,
    )

    if score > 0.5:
        result = "The model answered yes: it leaned toward false."
    elif score < 0.5:
        result = "The model answered no: it leaned away from false."
    else:
        result = "The model did not produce a clear yes/no answer."
    print(result)
    print(f"Yes score: {score:.2f}")


if __name__ == "__main__":
    main()
