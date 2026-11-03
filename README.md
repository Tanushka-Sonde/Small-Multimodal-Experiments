# Small Multimodal Experiments

Four hands-on experiments spanning contrastive vision-language alignment, frozen-embedding retrieval, vision-language model fine-tuning, and interpretable concept-based classification. Built for **Deep Learning** (Prof. Nipun Batra, IIT Gandhinagar, Assignment 3, October 2026).

## Overview

| # | Experiment | Core Idea |
|---|------------|-----------|
| Q1 | Contrastive CLIP Loss | Implementing the symmetric image-text contrastive loss from scratch |
| Q2 | Multimodal Retrieval | Text-to-image semantic search over a frozen embedding gallery |
| Q3 | VLM Adaptation | Fine-tuning only a vision-to-language connector for digit recognition |
| Q4 | Concept Bottleneck Models | Test-time intervention on interpretable concepts |

## Q1: Contrastive Image-Text Alignment

Implemented `clip_symmetric_loss` from scratch using L2-normalized embeddings, a temperature-scaled cosine similarity matrix, and `logsumexp`, without any pre-built contrastive loss function, over 4 paired image and text vectors.

- **Aligned pairs (τ=0.1):** loss = 0.000246, with the similarity matrix dominated by large diagonal entries (9.99)
- **Permuted text rows (τ=0.1):** loss jumps to 9.615, since the diagonal now holds mismatched pairs
- **Temperature sweep on aligned pairs:** loss falls from 0.771 at τ=1.0 to effectively 0 at τ=0.01, since a smaller τ sharpens the softmax and rewards the already-correct matches more aggressively

The diagonal of the similarity matrix always represents the true pairs by construction, which is why permuting only the text rows is enough to break alignment entirely.

## Q2: Multimodal Image Retrieval with Frozen Embeddings

Built `semantic_search`, a text-to-image retrieval function over a 17-image gallery, using frozen `google/embeddinggemma-2` embeddings and cosine similarity, with no re-encoding of the gallery per query.

- Two straightforward object queries (a rocket, a red mug) returned confident top hits around 0.70-0.75 similarity, each leading the runner-up by 0.06 to 0.14
- A fine-grained visual-detail query (an orange spacesuit) also resolved correctly to the right image
- A query for an object absent from the gallery (a purple unicorn in a sunflower field) still returned 5 results, but with scores packed into a flat 0.021-wide band and only 0.003 separating rank 1 from rank 2, showing the retrieval has no concept of "no match" and simply returns the least-bad candidates

## Q3: Vision-Language Model Adaptation

Adapted `HuggingFaceTB/SmolVLM-256M-Instruct` for handwritten digit recognition by fine-tuning only its **7,077,888-parameter** vision-to-language connector, with the vision encoder and language backbone fully frozen.

- **Setup:** 200 training images, 50 validation images, seed 42, AdamW at lr=2e-4, gradient clipping at 1.0, 100 steps, batch size 4
- **Before adaptation:** 76.0% accuracy, 100% valid single-digit responses
- **After adaptation:** 98.0% accuracy, 100% valid single-digit responses
- The errors fixed were genuine visual recognition mistakes (a 6 read as 0, a 9 read as 1, an 8 read as 2), not formatting issues, since the baseline model already answered in a single valid digit every time. Training only the connector reshapes how visual features map into the language model's input space, which was enough to fix most of these confusions in just 100 optimizer steps

Also implemented `infer`, separating a single diagnostic forward pass (returning logits of shape `(1, 80, 49280)`) from full autoregressive `generate` calls, confirming that the greedy next-token prediction from the forward pass is exactly the first token of the final generated answer.

## Q4: Concept Bottleneck Routing and Intervention

Implemented a strict Concept Bottleneck Model head (`cbm_head`) mapping 2 interpretable concepts, `is_red` and `is_circle`, to a binary "red-circle" prediction via a linear AND-gate (weights `[1.0, 1.0]`, bias `-1.5`, threshold `0.0`), plus a test-time `intervene` function.

- Before intervention, a misclassified `is_red` concept (`[0.0, 1.0]`) gave a score of **-0.5** and predicted "other"
- After correcting just `is_red` to 1.0, the score became **+0.5**, crossing the threshold and flipping the prediction to "red-circle"
- Because a strict CBM computes its label only from the concept vector, intervening on a concept directly and causally changes the output, something that is not possible in a standard end-to-end black-box classifier, where a post-hoc explanation is never read by the model and editing it changes nothing

## Repository Structure

```
small-multimodal-experiments/
├── assignment3.ipynb      # Full notebook: Q1-Q4, code, outputs, explanations
├── data/
│   ├── clip_vectors.json  # Paired image/text vectors for Q1
│   ├── gallery/           # 17-image gallery and manifest for Q2
│   ├── digits.npz         # Train/val/test handwritten digit splits for Q3
│   └── cbm.json           # Concept vectors and classifier head for Q4
└── README.md
```

## Tech Stack

PyTorch, Transformers, Sentence-Transformers, `google/embeddinggemma-2`, `HuggingFaceTB/SmolVLM-256M-Instruct`, NumPy, Matplotlib, PIL.

## Notes

All experiments use fixed seeds (42, 43, 44) and pinned model revisions for reproducibility. This was a group assignment; implementation, experiments, and write-ups were completed collaboratively, with AI tools used to check understanding and assist with plotting code.
