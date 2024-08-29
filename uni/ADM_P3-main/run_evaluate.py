from argparse import ArgumentParser
from time import time

import evaluate
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AlignProcessor, AlignModel


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="wds_vtab-caltech101",
        help="Dataset from (https://huggingface.co/clip-benchmark) to evaluate",
    )
    parser.add_argument(
        "--align_model",
        default="kakaobrain/align-base",
        help="ALIGN model from Huggingface.",
    )
    parser.add_argument(
        "--ensemble_methods",
        default=["baseline", "mean_input", "mean_logit", "mean_softmax"],
        nargs="+",
        help="Ensemble method for classification",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    dataset_id = f"clip-benchmark/{args.dataset}"
    dataset_dict = load_dataset(dataset_id)
    dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]])
    dataset = dataset.rename_columns({"webp": "image", "cls": "class"})
    classes = load_dataset(dataset_id, data_files="classnames.txt")["train"]["text"]
    templates = load_dataset(
        dataset_id, data_files="zeroshot_classification_templates.txt"
    )["train"]["text"]

    # Load model
    model_id = args.align_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlignModel.from_pretrained(model_id)
    model.to(device)
    processor = AlignProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load classifier
    for ensemble_method in args.ensemble_methods:
        print(f"\nUsing method {ensemble_method}")
        start = time()
        if ensemble_method == "baseline":
            inputs = tokenizer(classes, padding=True, return_tensors="pt")
            inputs = {k: t.to(model.device) for k, t in inputs.items()}
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)

            def zero_shot_classify(images, text_features):
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: t.to(model.device) for k, t in inputs.items()}
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # Cosine similarity and softmax with temperature
                image_features = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                logits = 100.0 * (image_features @ text_features.t())
                probs = logits.softmax(dim=1)
                pred = probs.argmax(dim=1)
                return pred

        elif ensemble_method == "mean_input":
            text_features = []
            for name in classes:
                # Compute mean for all templates
                name_templates = [template.format(c=name) for template in templates]
                inputs = tokenizer(name_templates, padding=True, return_tensors="pt")
                inputs = {k: t.to(model.device) for k, t in inputs.items()}
                with torch.no_grad():
                    name_text_features = model.get_text_features(**inputs)
                avg_text_features = name_text_features.mean(0)
                text_features.append(avg_text_features)
            text_features = torch.stack(text_features)

            def zero_shot_classify(images, text_features):
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: t.to(model.device) for k, t in inputs.items()}
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # Cosine similarity and softmax with temperature
                image_features = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                logits = 100.0 * (image_features @ text_features.t())
                probs = logits.softmax(dim=1)
                pred = probs.argmax(dim=1)
                return pred

        elif ensemble_method in ("mean_logit", "mean_softmax"):
            text_features = []
            for name in classes:
                name_templates = [template.format(c=name) for template in templates]
                inputs = tokenizer(name_templates, padding=True, return_tensors="pt")
                inputs = {k: t.to(model.device) for k, t in inputs.items()}
                with torch.no_grad():
                    name_text_features = model.get_text_features(**inputs)
                text_features.append(name_text_features)
            text_features = torch.stack(text_features)

            def zero_shot_classify(images, text_features):
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: t.to(model.device) for k, t in inputs.items()}
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                s = text_features.shape
                text_features_view = text_features.view((s[0] * s[1], s[2]))

                # Cosine similarity and softmax with temperature
                image_features = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_features_view = text_features_view / text_features_view.norm(
                    p=2, dim=-1, keepdim=True
                )
                logits = 100.0 * (image_features @ text_features_view.t())

                logits = logits.view((logits.shape[0], s[0], s[1]))

                if ensemble_method == "mean_logit":
                    probs = logits.mean(-1).softmax(dim=1)
                else:
                    probs = logits.softmax(dim=1).mean(-1)

                pred = probs.argmax(dim=1)

                return pred

        print(f"Using dataset {args.dataset}")
        print(f"With {len(classes)} classes and {len(templates)} templates")
        print(f"Setup time: {time()-start}")

        # Evaluate
        accuracy = evaluate.load("accuracy")
        for sample in tqdm(
            dataset.iter(batch_size=args.batch_size),
            total=len(dataset) / args.batch_size,
        ):
            pred = zero_shot_classify(sample["image"], text_features)
            accuracy.add_batch(references=sample["class"], predictions=pred)
        print(accuracy.compute())


if __name__ == "__main__":
    main()
