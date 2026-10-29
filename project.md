# Text-Guided Few-Shot Attribute Recognition

## Motivation

The task is person-level PPE attribute recognition.  Given a query person crop,
we predict attributes such as `Gown`, `Mask`, `Eyewear`, `Glove-L`, and
`Glove-R`.

The project starts from a human-like decision process:

1. Read the textual definitions of the candidate classes.
2. Form a mental concept for each class.
3. Look at a few reference examples for each class.
4. Compare the query image against both the textual concept and the visual
   examples.
5. Select the class with the highest evidence.

For example, for `Mask`, the model should distinguish:

- `NC`: N95 or blue/white mask fully covering nose and mouth.
- `RC`: light yellow or light purple mask fully covering nose and mouth.
- `MI`: mask covering the mouth but leaving the nose exposed.
- `MA`: absent or improperly worn mask.
- `NA`: face region, including the nose and mouth area, is occluded or outside
  the crop, making mask status indeterminable.

The main challenge is that each reference image is a full person crop, not a
crop of only the target attribute.  A `Mask=NC` reference crop still contains
gown, eyewear, gloves, body pose, background, and camera-view information.  A
naive average of whole-image embeddings can therefore encode irrelevant visual
factors.

## Baselines

We use SigLIP2 as the initial vision-language backbone.

Let:

- `q` be the query person crop.
- `c` be a candidate class.
- `d_c` be the textual description of class `c`.
- `R_c = {r_c,1, ..., r_c,K}` be the reference crops for class `c`.
- `I(.)` be the SigLIP2 image encoder.
- `T(.)` be the SigLIP2 text encoder.
- `cos(.)` be cosine similarity.

### Baseline A: Text-Only

This baseline compares the global query image embedding directly with each
class text embedding.

```text
score_text(c) = cos(I(q), T(d_c))
prediction = argmax_c score_text(c)
```

This tests whether SigLIP2 can solve the attribute using only class
descriptions.

### Baseline B: Global Reference Prototype

This baseline creates a visual prototype by averaging global SigLIP2 image
embeddings of the reference crops.

```text
p_c_global = mean_k I(r_c,k)
score_ref(c) = cos(I(q), p_c_global)
prediction = argmax_c score_ref(c)
```

This tests whether few-shot visual examples help without using text during
prototype construction.

### Baseline C: Text + Global Reference Fusion

This baseline combines Baseline A and Baseline B.

```text
score_fusion(c) =
    alpha * score_text(c)
  + beta  * score_ref(c)

prediction = argmax_c score_fusion(c)
```

The current default is:

```text
alpha = 0.5
beta = 0.5
```

This is useful as a simple reference-augmented baseline, but it still suffers
from the irrelevant-feature problem because the reference prototype is built
from whole-person embeddings.

## Proposed Method: Text-Guided Visual Prototype

The proposed method builds class prototypes by using the class description to
select relevant visual tokens inside each reference crop.

Instead of:

```text
mean_k I(r_c,k)
```

we use text-guided pooling over SigLIP2 vision tokens.

Let:

```text
V_i = [v_i,1, v_i,2, ..., v_i,N]
```

be the SigLIP2 vision tokens of reference image `r_c,i`, and:

```text
t_c = T(d_c)
```

be the text embedding of class `c`.

For each token, compute text-token affinity:

```text
a_i,j^c = softmax_j(cos(v_i,j, t_c) / tau)
```

Then construct a text-guided visual embedding for reference image `i`:

```text
g_i^c = sum_j a_i,j^c * v_i,j
```

The guided class prototype is:

```text
p_c_guided = mean_i g_i^c
```

For the query image, we apply the same text-guided pooling:

```text
g_q^c = guided_pool(q, d_c)
```

The proposed score is:

```text
score_guided(c) = cos(g_q^c, p_c_guided)
prediction = argmax_c score_guided(c)
```

Interpretation:

```text
score_guided(c)
= similarity(
    text-guided image query for class c,
    text-guided image prototype for class c
  )
```

The text embedding is not multiplied with the global image embedding.  It is
used as an attention/query signal to decide which image tokens should
contribute to the class-specific visual representation.

## Current Implementation

Main files:

- `src/attribute_recognition/backbones.py`
  - `SigLIP2Backbone`
  - global image encoding
  - text encoding
  - image-token encoding
  - text-guided image-token pooling

- `src/attribute_recognition/siglip2_classifier.py`
  - builds text prototypes
  - builds global reference prototypes
  - builds text-guided reference prototypes
  - predicts with `text`, `reference`, `fusion`, or `guided` mode

- `scripts/run_siglip2_attribute.py`
  - command-line entry point
  - supports single index, explicit indices, random sampling, label filters,
    and XLSX output

- `reference/{attribute}_{class}/`
  - stores 5 bbox-cropped reference person images per class

Example commands:

```bash
# Single query, guided mode
python scripts/run_siglip2_attribute.py \
  --data dataset/five_simulation/y1.json \
  --attribute Mask \
  --index 0 \
  --mode guided

# Random 10 non-NA Mask examples from y1.json
python scripts/run_siglip2_attribute.py \
  --data dataset/five_simulation/y1.json \
  --attribute Mask \
  --random 10 \
  --seed 13 \
  --exclude-label NA \
  --mode guided \
  --output-xlsx outputs/siglip2_scores/Mask_y1_nonNA_random10_seed13_guided.xlsx
```

## Output Columns

The XLSX output is long-format, one row per query/class pair.

- `score`: the score used for argmax in the selected mode.
- `text_score`: `cos(I(q), T(d_c))`.
- `reference_score`: `cos(I(q), p_c_global)`.
- `global_q_vs_guided_ref`: `cos(I(q), p_c_guided)`.
- `guided_q_vs_guided_ref`: `cos(g_q^c, p_c_guided)`.

For `--mode guided`:

```text
score = guided_q_vs_guided_ref
```

For `--mode text`:

```text
score = text_score
```

For `--mode reference`:

```text
score = reference_score
```

For `--mode fusion`:

```text
score = alpha * text_score + beta * reference_score
```

## Preliminary Observations

Experiments so far used `Mask` recognition on `dataset/five_simulation/y1.json`.

### Random 10 Queries, Mask GT != NA

Conditions:

```text
data = dataset/five_simulation/y1.json
attribute = Mask
exclude-label = NA
random = 10
seed = 13
indices = 1061,1158,854,974,738,958,857,660,324,923
```

Observed accuracy:

```text
text:      6/10
reference: 5/10
fusion:    5/10
guided:    5/10
```

Pattern:

- Text-only is stronger for `RC` and `NC` in this sample.
- Reference and guided modes are stronger for `MA`.
- `RC` is often confused with `MI` by reference/guided modes.
- Some `NC` examples are confused with `NA`.

### Mask NA Description Update

The original generic `NA` text was:

```text
The attribute cannot be determined because the relevant body part or equipment
is occluded or not visible.
```

This was changed to a mask-specific description:

```text
The face region, including the nose and mouth area, is occluded or outside the
crop, making mask status indeterminable.
```

For one random 10-sample Mask experiment, this improved guided accuracy from
`4/10` to `5/10`, mainly by correcting one `NA` case that was previously
classified as `NC`.

## Current Concerns

1. Whole-person reference crops contain strong irrelevant signals.
   - Gown, eyewear, glove, pose, and background can leak into mask prototypes.

2. Guided token pooling is conceptually better but still unstable.
   - Margins are often small.
   - Some guided predictions are near ties.

3. `NA` is hard.
   - `NA` means the evidence is absent or indeterminable.
   - Similarity-based matching tends to prefer visible visual patterns.
   - We may need special handling for uncertainty/visibility classes.

4. `RC` vs `MI` is weak in reference/guided modes.
   - The model may focus on mask color/face area but fail to distinguish nose
     coverage.

## Next Steps

1. Add DINOv3 visual reference scores.
   - Use DINOv3 only within its own visual embedding space.
   - Do not directly compare SigLIP2 and DINOv3 vectors.
   - Fuse scores at the scalar score level.

2. Add heatmap visualization.
   - Show which image tokens are selected by each class description.
   - Verify whether `Mask=NC`, `Mask=RC`, and `Mask=MI` attend to the nose/mouth
     region.

3. Improve class descriptions.
   - Make each attribute class more specific.
   - Avoid generic `NA` descriptions.
   - Write attribute-specific `NA` definitions.

4. Evaluate on larger samples.
   - Per-attribute accuracy.
   - Per-class confusion matrix.
   - Compare `text`, `reference`, `fusion`, and `guided`.

5. Consider special treatment for `NA`.
   - Use an evidence/visibility detector.
   - Predict `NA` when relevant region confidence is low.
   - Avoid treating `NA` as a normal positive visual class.
