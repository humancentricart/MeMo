# Saturation analysis of the MeMo correlation matrix memories

## 1. What is being measured

### 1.1 The memory and the recall operation

A correlation matrix memory stores pairs `(k_i, v_i)` as a sum of outer products:

```
M = Σ_i  k_i v_i^T                (here: CMM_OUT, a d × d matrix, d = 2048)
```

Recalling with a stored key gives the target plus interference from all other pairs ("crosstalk"):

```
M k_i  =  ‖k_i‖^2 · v_i   +   Σ_{j!=i} (k_j · k_i) · v_j
          └── signal ──┘      └────────── crosstalk ──────────┘
```

In the experiments, `k` is the composed key encoding of `input_sequence` (unit-norm, see `key_norm_mean`), `v` is the corresponding `output_symbols` vector, and a "pair" is one position selected by `to_save_sequences`. The recorded batches are replayed into an empty memory through `layer.memorize(...)`. At each checkpoint (a number of memorized batches) the evaluator measures how well stored keys recall their values. The last batches are never memorized and serve as unseen keys.

**Load.** `P` is the number of stored pairs and `α = P/d` is the load. The theory below shows that, for random keys, α alone determines recall quality.

### 1.2 The reference ("null") model

The theoretical reference model assumes:

1. Keys are independent, isotropically distributed, unit-norm vectors.
2. Values are independent of keys and have comparable squared norms.
3. The memory is initialized to zero and each association is stored through an outer-product update.
4. Crosstalk is approximately uncorrelated with the target signal.

Each key has inner product ≈ N(0, 1/d) with every other key. 

**The crosstalk power relative to the target is then `(P−1)/d ≈ α`.**

#### Details 
The fundamental question is: *how much two random unit-norm keys overlap?*

1. Overlap between two random keys. 
Take two independent, isotropic, unit-norm keys in d dimensions. 
Their inner product g = k_j · k_i has mean 0 and variance
```
E[g^2] = E [ (k_i^T k_j) (k_j^T k_i) ] = k_i^T E[k_j k_j^T] k_i = k_i^T (I/d) k_i = 1/d
```
The reason is that a random direction spreads its unit of energy evenly over d axes, so its squared overlap with any fixed direction is 1/d.

2. Interference from all the other pairs. 
Recalling with a stored key gives
```
M k_i = v_i + Σ_{j!=i} g_j v_j        (g_j = k_j · k_i, unit-norm keys)
```
The first term is the target. The second is a sum of P−1 independent terms, each with power ‖v_j‖^2 · 1/d. So the interference power is
```
E‖crosstalk‖^2 = (P−1) · ‖v‖^2 / d
```

3. Noise-to-signal ratio. Dividing by the signal power ‖v_i‖^2 ≈ ‖v‖^2 gives
```
noise / signal = (P−1)/d ≈ P/d  ≡  α
```
Each stored pair adds interference of relative power 1/d, and there are about P of them. 



### 1.3 Metrics under the reference model 

This gives the baseline that every metric is compared against:

| Quantity | Null-model prediction |
|---|---|
| Cosine between recall and target | `1/√(1+α)` |
| Relative L2 error `‖recall − v‖/‖v‖` | `√α` |
| held-out probe ratio, signal over crosstalk | `1/α`|
| Power of recall of stored keys / power of recall of unseen keys | `1 + d/P` |
| `‖M‖^2_F / (P · mean‖v‖^2)` | `1` |

#### Details

##### Cosine between recall and target

Definition:
```
(Mk_i)^T v_i / (|Mk_i||v_i|)
``` 
Let `c_i` be the crosstalk term. Since (`Mk_i=v_i+c_i`), and assuming that crosstalk is approximately orthogonal to the target:

```
(Mk_i)^T v_i ~ |v_i|^2
```

The expected squared norm of the recalled vector is:
```
E[|Mk_i|^2] = E[|v_i+c_i|^2] ~ |v_i|^2 + E[|c_i|^2] ~ (1+α)|v_i|^2
```

Substituting into the cosine expression gives:
```cos_theory = 1/sqrt(1+α)```

Interpretation: This metric measures the directional alignment between the recalled vector and its target. As memory load increases, crosstalk increases the magnitude of the recalled vector without contributing significantly in the target direction. Consequently, cosine similarity decreases.


##### Relative L2 reconstruction error

Definition:
```
(|Mk_i-v_i|)/|v_i|
```

Since `Mk_i=v_i+c_i`: `Mk_i-v_i=c_i`. Therefore the reconstruction error is 
```
|c_i|^2 / |v_i|^2
```

Taking expectations:

```
E[|c_i|^2 / |v_i|^2] ~ (P-1)/d ~ α
```
So the expected relative L2 reconstruction error is 
```rel_err_theory = sqrt(α)```

Interpretation: This metric measures the magnitude of the reconstruction error relative to the target magnitude. A value of zero indicates perfect recall, a value of one indicates an error with the same magnitude as the target, and values greater than one indicate that the error exceeds the target magnitude.

Unlike cosine similarity, relative L2 error measures the magnitude of the deviation rather than its angular effect.

##### Held-out probe signal-to-crosstalk ratio

A held-out probe is an unseen key (k_*) that has not been stored in the memory.

Retrieving an unseen key gives:
```
Mk_* = Σ_{j=1}^{P}(k_j^T k_*)v_j
```

Since there is no corresponding stored target, all contributions are interference.

Under the null model:

```
E[|Mk_*|^2] ~ (P/d)|v|^2 = α|v|^2
```

For a stored key, the crosstalk power is approximately:
```
E[|c_i|^2]~ α|v|^2
```

The signal-to-crosstalk ratio for a stored association is therefore:
```
|v_i|^2/ E[|c_i|^2] = 1 / α
```

So the expected held-out probe signal-to-crosstalk ratio is:
``` 1 / sqrt(α)```

Interpretation: This metric measures the strength of the target signal relative to interference from other memories. It decreases as more associations are stored, indicating that the target becomes progressively harder to distinguish from crosstalk.

Note that this metric is specifically the signal power divided by crosstalk power. It is not the ratio between the total recall power of stored keys and unseen keys, which is measured separately below.

##### Ratio of stored-key recall power to unseen-key recall power
Definition:
`‖M k_stored‖^2 / ‖M k_unseen‖^2`
This metric compares the average squared magnitude of the memory response to stored keys against its response to unseen keys.

For a stored key: `Mk_i=v_i+c_i`

Assuming signal and crosstalk are approximately uncorrelated:

```
E[|Mk_i|^2] ~ (1+α)|v|^2
```

For an unseen key: `Mk_=Σ_{j=1}^{P}(k_j^T k_)v_j`

Its expected squared recall magnitude is:
```
E[|Mk_*|^2] ~ α|v|^2
```

The rate is therefore:
```
power_ratio = E[|Mk_i|^2] / E[|Mk_*|^2] = ((1+α)|v|^2) / (α|v|^2) = 1+ 1/α = 1 + d/P
``` 

Interpretation: Stored keys receive an additional contribution from their own associated values, whereas unseen keys receive only interference. This ratio measures how distinguishable stored keys are from unseen keys based on the magnitude of their recall responses.

As memory load increases, the ratio approaches one, indicating that the recall magnitude of stored keys becomes increasingly similar to that of unseen keys.


##### Normalized squared Frobenius norm
This metric measures the total energy accumulated in the memory matrix rather than recall quality directly.

Definition:
```
Q = (|M|_F^2) / (P E[|v|^2])
```

Recall that: `M=\sum_{i=1}^{P}k_i v_i^T`

Expanding the squared Frobenius norm: 
```
|M|F^2 = | \sum{i=1}^{P}k_i v_i^T |F^2 = 
\sum{i=1}^{P} \sum_{j=1}^{P} dot(k_i v_i^T, k_j v_j^T)_F = 
\sum_{i=1}^{P}\sum_{j=1}^{P} (k_i^T k_j)(v_i^T v_j)
```

We can separate the diagonal and off-diagonal terms.

For the diagonal terms ((i=j)), since keys are unit-normalized:
```
|v_i|^2
```

Their total contribution is:
```
\sum_{i=1}^{P}|v_i|^2
```

For the off-diagonal terms ((i\neq j)), independent isotropic keys have zero expected inner product:
```
E[k_i^T k_j]=0
```

Assuming keys and values are independent, the expected off-diagonal contribution is also zero.

Consequently:
```
E[|M|F^2] = \sum_{i=1}^{P} E[|v_i|^2] = P E[|v|^2]
```

The normalized prediction is therefore:

`fro2_per_pair = Q = (|M|_F^2) / (P E[|v|^2]) ~ 1` 


Interpretation: Under the null model, every stored association contributes approximately its own value energy to the memory matrix. Cross-terms cancel out in expectation, so the normalized Frobenius norm should remain close to one regardless of memory load.

Deviations from one can indicate correlations between keys, interactions between key and value statistics, non-unit key norms, or differences in the memory update or normalization procedure.

A value below one is also possible in finite samples because cross-terms can partially cancel the diagonal contributions.

The value of the null model is diagnostic: agreement means the memory behaves as efficiently as a linear associator can, and deviations point to structure (correlated keys, repeated pairs, heterogeneous values) that costs or adds capacity.


#### Estimated effective key dimension
Suppose that keys effectively occupy `d_eff` independent dimensions.

The expected squared overlap between two independent unit-norm keys is approximately:
```
E[(k_i^Tk_j)^2] ~ 1 / d_eff
```

There are `P-1` other stored associations contributing to the crosstalk.

Consequently:
```
E[|c|^2] ~ (P-1) /(d_eff) |v|^2
```

We aim to estimate:
```
d_eff ~ ((P-1) |v|^2)/E[|c|^2] 
```

We can sunstitute the `|v|^2 /E[|c|^2]` with the `power_ratio` as follows.

The `power_ratio` compares the average squared recall magnitude for stored keys against unseen keys: `E[|Mk_i|^2] / E[|Mk_*|^2]`

Assuming signal and crosstalk are approximately uncorrelated:
```
E[ |Mk_i|^2 ] ~ |v|^2+E[|c_i|^2]
```

For an unseen key, there is no signal:
```
E[|Mk_*|^2] ~ E[|c_*|^2]
```

Under the assumption that stored and unseen keys are sampled from the same distribution, their expected crosstalk powers are approximately equal.
Therefore:
```
power_ratio ~ (|v|^2+E[|c|^2]) / (E[|c|^2])
```
Rearranging:
```
power_ratio-1 ~ (|v|^2)/ (E[|c|^2])
```
In other words, `power_ratio-1` isolates the signal-to-crosstalk ratio.


Substituting above, we have:
```
d_eff ~ (P-1) power_ratio
```



### 1.3 Metric reference

| Metric | What it computes | Why it matters | How to read it |
|---|---|---|---|
| `P`, `alpha` | Number of stored pairs; `P/d` | The control variable. Capacity scales with `d`, so results are comparable across layers only at equal α. | Compare layers at equal α, not equal number of batches, because pairs per batch differ (§2). |
| `cos_mean` | Mean cosine between `M k_i` and `v_i` over a random sample of stored pairs | Scale-free fidelity of raw recall. `cos^2` is the fraction of the recalled vector's energy that lies along the target. | 0.9 means almost clean; 0.5 means 25 % of the energy is on target; near 0 means the recall is mostly noise. |
| `cos_theory` | `1/√(1+α)` | Null-model reference. | Measured ≈ theory means keys behave as random vectors. |
| `cos_p05`, `cos_min` | 5th percentile and minimum of the per-pair cosine | Tail behaviour: the mean can hide pairs that are recalled badly. | A large gap between mean and p05 means pairs are not exchangeable. |
| `rel_err` | Mean of `‖M k_i − v_i‖ / ‖v_i‖` | Magnitude of the error, not just its direction. A recall of zero has `rel_err = 1`. | Above 1 means the raw recall is further from the target than an empty answer; direction can still be informative. |
| `rel_err_theory` | `√α` | Reference for the above. | Excess over theory means the recall has extra non-target energy. |
| `top1_acc` | Fraction of recalls whose true value is the nearest (cosine) among all `P` stored values. Ties by similarity count as correct, so duplicate values are handled. | Models a cleanup / decoding stage. Measures how much *information* survives, not how clean the vector is. | The bank grows with `P`, so the task gets harder with load. Compare to `top1_chance = 1/P`. |
| `key_norm_mean` | Mean norm of the encoded keys | Sanity check that the unit-norm assumption of the theory holds. | 1.0 in all layers, so the theory applies as is. |
| `power_ratio` | `‖M k_stored‖^2 / ‖M k_unseen‖^2` | Unseen keys have no signal, so their recall is pure crosstalk. The ratio of stored to unseen power gives signal-to-crosstalk without needing targets. | Null model: `1 + d/P`. |
| `d_eff_est` | `P · (power_ratio − 1)` | Estimated effective key dimension, the "room" the memory really has. | ≈ `d` means keys use all dimensions; much below `d` means correlated or low-rank keys; above `d` is not explained by key anisotropy. |
| `fro2_per_pair` | `‖M‖^2_F / (P · mean‖v‖^2)` | Equals `1 + (1/(P·mean‖v‖^2)) Σ_{i!=j} (k_i·k_j)(v_i·v_j)`. Detects whether similar keys go with similar values. | ≈ 1: no systematic relation. > 1: similar keys carry similar values (redundant or consistent associations). < 1: similar keys carry opposite values (conflicts). |



#### 1.4 Further analysis worth reading

- Bound on the top-1 with high probability  https://arxiv.org/pdf/2605.05189
- Analysis of feature correlations https://arxiv.org/html/2508.01395v1
- Thesis with (maybe) hints on generalization: https://etheses.whiterose.ac.uk/id/eprint/2211/1/FinalSubmission.pdf


## 2. Decoding: from the raw recall to a probability over values


### 2.1 Why a decoding stage is needed

The metrics of §1 (cosine, relative error, `d_eff`, ...) describe the *vector* `r = M k`. They do not say what a downstream consumer can do with it. In the model the values are not arbitrary vectors: each stored value is one of a finite vocabulary of **candidates** `u_1, ..., u_V` (the distinct value vectors found in the store, identified by exact equality; embeddings of tokens or symbols), and the recall is then turned into a **softmax distribution over candidates**. Two things matter:

1. **Ranking** (which candidate is most likely): measured by top-k accuracy.
2. **Calibration** (are the probabilities right?): measured by the negative log-likelihood (NLL) of the true candidate. A decoder can have the right arg-max and still give terrible probabilities if its temperature is wrong. Since the softmax output is what is used (for sampling, for the loss, for combining with other layers), calibration matters as much as ranking.

So the question is: *given the null model of §1.2, what is the best possible decoder, and how far is a plain `softmax(U r)` from it?* The answer is the Bayesian posterior derived below. All decoders in the evaluation are special cases or approximations of it.

### 2.2 Setting and notation

```
U ∈ R^{V×d}       candidate matrix, row c is u_c (distinct value vectors in the store)
c_j               candidate id of the value of stored pair j,   v_j = u_{c_j}
count_c           number of stored pairs whose value is candidate c   (Σ_c count_c = P)
π̂_c = count_c/P   empirical frequency of candidate c among the stored pairs
t                 candidate id of the target of the probe (the thing to be decoded)
r = M k_i         recall; k_i is a stored key whose true value is u_t
γ = ‖k_i‖^2       squared key norm (≈ 1; key_norm_mean is 1 in all layers)
α = (P−1)/d       load seen by one probe (the P−1 *other* pairs create the noise)
```

The values are **not** assumed to be isotropic or uniformly distributed over candidates. The empirical frequencies `π̂_c` are the data (e.g. Zipf-like for tokens: `π_c ∝ c^{-s}`, a few candidates account for most of the pairs).

### 2.3 The null model for the recall, and where `C` comes from

This is the same computation as §1.2, but keeping the *shape* of the noise instead of only its power.

**Step 1: write the recall as signal plus a sum over the other pairs.**

```
r = M k_i = γ u_t + n,        n = Σ_{j≠i} g_j v_j,        g_j = k_j · k_i
```

**Step 2: group the crosstalk by candidate.** Many pairs share the same value vector (that is what a vocabulary is). Collecting them:

```
n = Σ_c ξ_c u_c,        ξ_c = Σ_{j≠i : c_j = c} g_j
```

Each `g_j` is approximately `N(0, 1/d)` (§1.2, point 1), with zero mean because the key distribution is symmetric, and the `g_j` of different pairs are approximately independent given `k_i`. Therefore, with `n_c = count_c` (minus 1 for `c = t`, because the pair `i` itself is the signal):

```
ξ_c ~ N(0, n_c/d),   independent across c
```

**Step 3: noise distribution.** `n` is a linear combination, with independent Gaussian coefficients, of the candidate vectors. Hence `n` is Gaussian with mean zero (no central limit theorem over tokens is needed; it only requires the `g_j` to be Gaussian, which holds for large d). Its covariance is

```
Cov(n) = Σ_c (n_c / d) u_c u_c^T  =  ((P−1)/d) · Σ_c (n_c/(P−1)) u_c u_c^T   ≈   α · C

C  =  Σ_c π̂_c u_c u_c^T  =  (1/P) Σ_j v_j v_j^T  =  U^T diag(π̂) U            (d × d)
```

So **`C` is the second-moment matrix of the stored values**, and it is computed *exactly* from the stored counts (this is what the whitening code does, in float64, from the candidate matrix and `count_c`):

* It is a *second moment*, not a covariance: the mean of the values is not subtracted. The crosstalk coefficients `g_j` have zero mean, so what enters the noise is `E[n n^T] = (1/d) Σ_j v_j v_j^T`, mean included.
* It depends on the checkpoint: counts are those of the batches memorized so far, so `C` changes (slowly) along the sweep.
* The scalar noise power of §1.2 is its trace: `E‖n‖^2 = α · tr(C) = α · mean‖v‖^2`, which is `E‖crosstalk‖^2 = (P−1)‖v‖^2/d` again.
* `C` has rank at most `min(#distinct stored values, d)`. If values are Zipf-distributed, its spectrum is dominated by the few most frequent candidates. A convenient summary is the participation ratio `PR(C) = tr(C)^2 / ‖C‖_F^2` (stored as `C_participation_ratio`): it equals `d` if `C = σ²I` and is small if the energy sits in a few directions. If the most frequent candidates are nearly orthogonal, `PR(C) ≈ 1/Σ_c π̂_c^2`. For an illustration, Zipf(s=1) over 5·10⁴ candidates gives `Σ π² ≈ ζ(2)/H_V² ≈ 0.0127`, i.e. `PR ≈ 80`, far below `d = 2048`. This is a **statement about the values**, not about the keys: `d_eff` (§1.3) measures key isotropy, `C` measures value isotropy. They are independent diagnostics.

**Special case (the original decoders).** If values are isotropic with `C = σ_v² I`, `σ_v² = mean‖v‖²/d`, the noise is `N(0, s² I)` with `s² = α σ_v²`. This is exactly `s² = ((P−1)/d) σ_v²` in the isotropic Bayes decoder.

**What the model ignores** (so that its failures can be recognised later):
(i) the pairs that share the value of the target add extra noise along `u_t` itself, with variance `(count_t − 1)/d`, which is not small for very frequent candidates; (ii) keys are not exactly isotropic (see `d_eff_est`); (iii) the noise is treated as independent of the candidate being tested, so the `log det` terms, which would depend on `c`, are dropped.

### 2.4 The posterior over candidates

Under the model `r | (target = c) ~ N(γ u_c, α C)` and prior `π_c`, Bayes' rule gives `log p(c|r) = log p(r|c) + log π_c − log p(r)`. Expanding the Gaussian:

```
log p(r | c) = −1/(2α) (r − γ u_c)^T C^{-1} (r − γ u_c) + const
             = −1/(2α) [ r^T C^{-1} r − 2γ u_c^T C^{-1} r + γ² u_c^T C^{-1} u_c ] + const
```

The term `r^T C^{-1} r` and the `log det(2π α C)` do not depend on `c`, so they are absorbed by the softmax normalization. What remains is

```
log p(c | r)  =  (γ/α) · [ u_c^T C^{-1} r  −  (γ/2) · u_c^T C^{-1} u_c ]  +  log π_c  −  log Z(r)
```

With `γ = 1` this is the formula `(1/α)[u_c^T C^{-1} r − ½ u_c^T C^{-1} u_c] + log π_c`. In the code the decoder logits are

```
logit_c = β_w · ( u_c^T C_λ^{-1} r − ½ γ u_c^T C_λ^{-1} u_c ) + log π_c,       β_w = γ/α,
C_λ = C + λ (tr C / d) I          (ridge, parameter `ridge` = λ)
```

and the probability is `softmax_c(logit_c)`. Four ingredients appear, and each decoder in the evaluation switches some of them on.

### 2.5 The four ingredients

**(1) Temperature `1/β`: the noise level, not a constant.** The factor in front of the bracket is `γ/α`, so the softmax scale shrinks as the load grows. For unit-norm isotropic values (`σ_v² = 1/d`, so `C^{-1} = d·I`), the equivalent coefficient on the plain dot product `u_c·r` is

```
β = γ d / α = d² / (P−1)
```

| α | 0.5 | 13 | 149 |
|---|---|---|---|
| β (d = 2048) | ≈ 4100 | ≈ 160 | ≈ 14 |

A fixed `softmax(U r)` uses `β = 1`, which is the right value only at `α = d` (P ≈ d², here 4·10⁶ pairs). At lower loads, which are the interesting ones, it is far too flat (underconfident); beyond that it would be overconfident. The arg-max is unaffected, but NLL and every probability-weighted use of the output is.

*Where the evidence stands.* For orthogonal unit-norm candidates, the Bayes logit difference between the true and a wrong candidate is `β(γ + (u_t − u_w)·n)`, i.e. Gaussian with

```
mean = d/α,       standard deviation = sqrt(2d/α)          (nats)
```

So the evidence in favour of the truth is `d/α` nats (4096 at α = 0.5, 13.7 at α = 149) and the question is how it compares with (a) the number of competitors and (b) the prior's log-odds (below).

**(2) Whitening with `C^{-1}`.** With `C^{-1/2}` applied to both recall and candidates, the noise becomes isotropic with variance `α`:

```
r' = C^{-1/2} r,   u'_c = C^{-1/2} u_c      ⇒      n' ~ N(0, α I)
logit_c = (γ/α)[ u'_c · r' − ½ γ ‖u'_c‖² ]  =  −(1/2α) ‖r' − γ u'_c‖² + const
```

i.e. the decoder is a nearest-neighbour search in **Mahalanobis distance**. The plain dot product treats all directions as equally noisy. If values are clustered, low-rank or Zipf-distributed, the crosstalk is concentrated in the directions of the frequent candidates (large eigenvalues of `C`), and the plain dot product gives those noisy directions as much weight as the quiet, informative ones.

*A concrete picture* (idealized): suppose the candidates are orthonormal. Then `C = Σ π̂_c u_c u_c^T`, `C^{-1} u_c = u_c/π̂_c`, and the whitened logit is

```
logit_c = (r_c − ½γ) · γ / (α π̂_c) + log π̂_c,        r_c = u_c · r
```

In this "channel" the noise on `r_c` has variance `α π̂_c`. So **each candidate has its own temperature `1/(α π̂_c)`**: frequent candidates have a noisy channel (flat logits), rare candidates have a quiet one (sharp logits). A single global temperature, even a perfectly fitted one, cannot represent this, and this is the part of the improvement that fitting `β` alone cannot capture. (The idealization needs `V ≤ d`; with `V > d` the geometry is more complicated but the mechanism is the same.)

*Ridge.* If the number of distinct stored values is below `d`, `C` is singular: any direction outside the span of the stored values has zero noise, so a candidate with a component there would be assigned zero probability by the model. The ridge `λ (tr C/d) I` (relative to the mean eigenvalue) regularizes this and acts as an isotropic noise floor. It is a hyper-parameter: too small amplifies numerical noise, too large reverts to the plain decoder. Choose it by looking at the NLL on held-out probes.

**(3) The bias `½ γ u_c^T C^{-1} u_c`.** It makes the score `−½ ‖r − γ u_c‖²_{C^{-1}}` instead of an inner product. It matters only when the candidates have **different norms** (typical for token embeddings; often frequency-related): without it, the dot product favours large-norm candidates, independent of the evidence. With equal norms the term is constant and drops out of the ranking.

Putting the geometry side by side:

| Rule | Score | Nearest neighbour in |
|---|---|---|
| `dot_T1` | `u_c · r` | maximum inner product |
| cosine cleanup (`top1_acc`) | `u_c · r / (‖u_c‖‖r‖)` | angle |
| Bayes (`bayes_T`) | `u_c · r − ½γ‖u_c‖²` | Euclidean distance |
| whitened (`white_T`) | `u_c^T C^{-1} r − ½γ u_c^T C^{-1} u_c` | Mahalanobis distance |

They coincide only when norms are equal (first three) and values are isotropic (last one).

**(4) The prior `log π_c`.** It is free, since the stored counts are known. When the evidence is weak it carries a lot of weight. For Zipf(s) the log-odds between the most and the least frequent candidate are about `s · ln V` (≈ 10.8 nats for s = 1, V = 5·10⁴), to be compared with the evidence `d/α`: the prior dominates roughly when `d/α ≲ s ln V`, i.e. `α ≳ d/10` here. A decoder without prior wastes this information; a prior-only decoder is the correct fall-back at very high load. Note one consequence for the temperature: *without* a prior, `β` multiplies all logits and cannot change the arg-max; *with* a prior, `β` sets the balance between evidence and prior, so it can change the ranking (and top-k).

**If values were isotropic with equal norms,** (2) and (3) vanish and only the temperature and the prior remain. This is the reason the five original decoders form a ladder: `dot_T1` → add the bias and the theoretical temperature → add the prior → fit the temperature. The four whitened decoders repeat the ladder with `C^{-1}` in place of the identity.

### 2.6 The decoders in the evaluation

| Name | Logit | Temperature | Notes |
|---|---|---|---|
| `dot_T1` | `u_c·r` | fixed, 1 | plain unembedding (the "current" decoder) |
| `bayes_T` | `β (u_c·r − ½γ‖u_c‖²)` | `β = γ/s²`, theory | `s² = ((P−1)/d) σ_v²`, `σ_v² = mean‖v‖²/d` |
| `bayes_T_prior` | `bayes_T + log π_c` | theory | |
| `fit_T` | as `bayes_T` | fitted `β` | |
| `fit_T_prior` | as `bayes_T_prior` | fitted `β` | |
| `white_T` | `β_w (u_c^T C_λ^{-1} r − ½γ u_c^T C_λ^{-1} u_c)` | `β_w = γ/α`, theory | needs `ridge` |
| `white_T_prior` | `white_T + log π_c` | theory | |
| `fit_white_T` | as `white_T` | fitted `β_w` | |
| `fit_white_T_prior` | as `white_T_prior` | fitted `β_w` | |

`π_c = (count_c + a)/(P + aV)` with `a = prior_smoothing` (Laplace smoothing for `a = 1`; as `a → 0` the decoder is restricted to candidates actually stored). `V` includes candidates that appear only in the held-out batches, which are unseen at memorization time (they get the smoothing mass only).

**Fitting the temperature.** The probes are split in two folds. For each fold, `β` is chosen by grid search (21 values, ±3 octaves around the theoretical value) to minimize the NLL on that fold, and the decoder is scored on the *other* fold (2-fold cross-fitting, so the NLL is not optimistic for the one fitted parameter). The reported `beta_fit` is the geometric mean of the two fits. Fits are done separately with and without the prior, and for the plain and whitened scores.

`β_fit/β_theory` is the main diagnostic of the null model:

| `β_fit/β_th` | Reading |
|---|---|
| ≈ 1 | The Gaussian null model describes the crosstalk |
| < 1 | More noise than predicted: keys correlated (`d_eff < d`), pairs sharing values, heavier-than-Gaussian tails. In the isotropic case roughly `β_fit/β_th ≈ d_eff/d` |
| > 1 | Less noise than predicted (rare; e.g. strongly structured values that are easy to separate) |
| exactly `2^{±3}` | The grid edge was hit: the NLL is still improving. Typical at very small `P`, where the decoder is almost perfect and the NLL keeps improving as `β → ∞` |

### 2.7 Metrics and their expected behaviour

**NLL (nats).** For one probe, `NLL = −log softmax(logits)[t] = LSE(logits) − logit_t`; the reported number is its mean over probes. References:

```
uniform decoder:        NLL = ln V
prior-only decoder:     NLL = −mean ln π_t   ≈ H(π̂)    (entropy of the stored-value distribution)
perfect recall:         NLL → 0
```

For Zipf(1) over V = 5·10⁴: `ln V ≈ 10.8` nats while `H(π̂) ≈ ln H_V + (ln V)²/(2 H_V) ≈ 2.43 + 5.13 ≈ 7.6` nats. That gap is available to any prior-aware decoder for free. A calibrated decoder sits between 0 (low load) and the baseline it falls back on (high load: `ln V` without prior, `H(π̂)` with prior).

**Information from memory** `= NLL(prior only) − NLL(decoder)`. Since cross-entropy upper-bounds conditional entropy, this is (up to the in-sample prior) a **lower bound on the mutual information** `I(target; recall)` in nats. It is 0 when the recall is useless and bounded above by `H(π̂)` (the whole information content of a stored value).

**Top-k accuracy.** The fraction of probes whose true candidate id (matched by exact id, not by similarity) is among the k highest-scoring candidates. By construction it is non-decreasing in k and equals 1 for `k ≥ V`. Reference for the prior-only decoder: the mass of the `k` most frequent candidates, e.g. for Zipf(1) over V = 5·10⁴: `H_k/H_V` = 0.20 (k = 5), 0.26 (k = 10), 0.46 (k = 100). A decoder that uses the prior cannot fall below this at high load; one that does not use it falls to `k/V`.

*Where accuracy breaks down (isotropic approximation).* Wrong candidates receive noise of standard deviation `sqrt(α/d)` on a signal of 1 (unit-norm, near-orthogonal candidates), so top-1 is correct when `1 + s z_0 > s z_c` for all wrong `c`, with `s = sqrt(α/d)`:

```
P(top-1 correct) ≈ ∫ φ(z) Φ(z + sqrt(d/α))^{V−1} dz
```

and the transition happens when `d/α ≈ 2 ln V`, i.e.

```
α_1 ≈ d / (2 ln V)          and for top-k:   α_k ≈ d / (2 ln(V/k))
```

For d = 2048 and V = 5·10⁴: `α_1 ≈ 95`, `α_5 ≈ 111`, `α_10 ≈ 120`, `α_100 ≈ 165` (P ≈ 2·10⁵ to 3.4·10⁵). So the curves for larger k are shifted to the right, and the spacing between them is a direct reading of how sharply the memory degrades. These are order-of-magnitude predictions for the plain-geometry decoders; whitening and the prior should push the transition to the right, and heterogeneous pairs (Zipf) smooth it.

### 2.8 The plots

Each figure has two rows. Row 1 describes the raw recall; row 2 compares decoders. The x-axis is always `P` (log scale); to compare layers or runs convert to `α = P/d`.

#### Row 1, panel 1: cosine(recall, target)

Math: §1.3. Curves: measured mean, measured 5th percentile, theory `1/√(1+α)`.

| α | 0.1 | 0.5 | 1 | 13 | 149 |
|---|---|---|---|---|---|
| theory | 0.95 | 0.82 | 0.71 | 0.27 | 0.08 |

Interpretation: mean on the theory line means the crosstalk is white-noise-like and values have comparable norms. The p05 curve should hug the mean (the cosine concentrates for large d). With Zipf-distributed values, pairs are not exchangeable: the noise-to-signal ratio of pair `i` is `α · mean‖v‖²/‖v_i‖²`, so pairs whose value has a small norm are hurt more, and frequent candidates additionally suffer from the extra noise along themselves (§2.3, (i)). Both widen the mean-p05 gap and pull the mean away from theory.

#### Row 1, panel 2: top-1 cleanup (cosine, stored bank) vs chance

Math: for each probe, the true value is counted as correct if its cosine with `r` is not lower than the cosine of any value in the bank of all `P` stored values (ties within 10⁻⁴ count as correct, so duplicate values are handled). It is a nearest-neighbour search in *angle* over *stored pairs*; the decoders of row 2 instead search *Euclidean/dot/Mahalanobis* over *distinct candidates* with exact-id matching. Same idea, different geometry and different bank, so the numbers are not interchangeable.

Expected value: same transition as in §2.7 with `N` = number of distinct stored values in place of `V`: accuracy is ≈ 1 for `α ≪ d/(2 ln N)` and decays after. Note that the plotted "chance" `1/P` is the chance of an arbitrary bank *entry*. With repeated values (Zipf) the chance of picking an entry with the correct value is `count_t/P`, whose average over probes is `Σ_c π̂_c² ≈ 1/PR` (≈ 0.013 in the Zipf example), orders of magnitude above `1/P`. A stronger and fairer reference is the most-frequent-value accuracy `max_c π̂_c`.

#### Row 1, panel 3: `d_eff` estimate

Math: §1.3, `d_eff ≈ P · (power_ratio − 1)` (the derivation gives `(P−1)(power_ratio − 1)`; the difference is negligible for large P). Expected: flat at `d = 2048` for isotropic, uncorrelated keys. Below `d`: keys concentrate on fewer directions (correlated, low-rank), and everything in §3 gets worse by the same factor. At small `P`, `power_ratio − 1` is large and noisy, so only large-`P` points are reliable.

#### Row 2, panel (a): NLL of the true value

Math: §2.7. Curves: the nine decoders; `uniform` = `ln V` (gray dotted); `prior only` (black dashed); the shaded band is "information from memory" (between the prior-only line and the best decoder's curve).

Expected:

* All decoders start near 0 at small `P` (recall almost perfect; theory: signal-to-noise `d/α` in the thousands).
* The decoders with a fitted or theoretical temperature rise smoothly and saturate at `ln V` (no prior) or `H(π̂)` (with prior). The prior curves lie below the others, and the gap between the two plateaus, `ln V − H(π̂)`, is the entire value of the prior (≈ 3 nats in the Zipf(1) example).
* `dot_T1` (β = 1) is far from the others: for unit-norm values it is too flat at `α < d` (NLL close to `ln V`, at small α it never becomes sharp) and only reaches calibration at `α ≈ d`. A large gap between `dot_T1` and `fit_T` at low `α` is **pure miscalibration** (the arg-max is the same).
* Ordering to expect when the model is right: `fit_white_T_prior ≤ fit_T_prior ≤ fit_T`, `fit_white_T_prior ≤ white_T_prior` and, for each pair, theory ≥ fitted up to a small cross-fitting error. If `bayes_T ≈ fit_T`, the temperature theory is correct. If `white_* ≈ plain` the values behave as isotropic (check `C_participation_ratio` ≈ `d`).
* If the whitened curves are *worse* than the plain ones, suspect the ridge (too small), too few stored pairs for `C` to be well determined, or a mis-specified noise model.

#### Row 2, panel (b): accuracy of the decoders

Math: §2.7 (top-k, id matching). The random baseline is not drawn (it lies orders of magnitude below, `1/V`); the y-axis is linear so differences near 1 are visible.

Expected:

* Top-1 of `dot_T1` equals that of `bayes_T`/`fit_T` when candidate norms are equal (the bias only reorders when norms differ), and the top-k lines for the no-prior decoders do not depend on `β`.
* Prior decoders (`*_prior`) are higher at high load and equal at low load (when evidence ≫ prior). The same for whitened vs plain: they coincide for isotropic values and separate for Zipf/clustered values.
* All curves decrease with `P`; the transition (when accuracy leaves 1) at `α_k ≈ d/(2 ln(V/k))`; top-100 > top-10 > top-5 > top-1 at every `P`, and the gaps are larger in the transition region.
* At very high load every prior-aware curve should settle above the prior-only top-k (`H_k/H_V` for Zipf(1)); curves without a prior fall to `k/V`.
* The dotted black line (top-1 cosine cleanup) is a different decoder (angle, stored bank, tie-tolerant) and is a reference, not a bound.

#### Row 2, panel (c): inverse temperature

Math: `β_theory = γ/s² = γ d² / ((P−1) mean‖v‖²)` for the plain decoders and `β_w = γ/α = γ d/(P−1)` for the whitened decoders; fitted curves are the geometric mean of the two cross-fitting folds, with and without prior.

Expected: straight lines of **slope −1** in log-log (β ∝ 1/P), e.g. `β ≈ d²/(P−1)` for unit-norm isotropic values: 42 at P = 10⁵, 4.2 at P = 10⁶. Fitted points on top of the theory line mean the Gaussian null model is valid; a parallel offset below means more noise than predicted (see §2.6 table). Points stuck at `β_th · 2^{±3}` mean the grid edge was hit. Because `β_w` is defined with respect to `C^{-1}` and `β_theory` with respect to the identity, **the two families live on different scales** (their ratio is `1/σ_v²` ≈ `d/mean‖v‖²`), so the absolute values are not comparable between them. Compare each fitted curve with *its own* theoretical curve, or plot the ratio `β_fit/β_theory`, which is also printed by `sweep`.

### 2.9 Expected values at a glance

| Quantity | Isotropic unit-norm reference | With Zipf-distributed values |
|---|---|---|
| noise covariance of recall | `α σ_v² I` | `α C`, `C = Σ π̂_c u_c u_c^T` |
| equivalent dot-product temperature `β` | `d²/(P−1)` | per-direction: `1/(α · eigenvalue of C)` |
| evidence per wrong candidate (nats) | mean `d/α`, sd `√(2d/α)` | candidate-dependent (`∝ 1/(α π̂_c)`) |
| NLL plateau at high load | `ln V` | `H(π̂)` with prior; `ln V` without |
| top-k plateau at high load | `k/V` | `Σ_{c≤k} π̂_c` (with prior) |
| `α` where top-k fails | `d/(2 ln(V/k))` | later (prior + whitening) |
| `β_fit/β_theory` | 1 | 1 if the whitened model is right; plain family < 1 |
| `C_participation_ratio` | `d` | ≈ `1/Σ π̂²` (tens, for Zipf(1)) |
| info from memory | `ln V` at α → 0 | `H(π̂)` at α → 0; 0 at α → ∞ |

### 2.10 A model-validation check worth adding

The Gaussian model predicts the squared *whitened* residual of a stored probe:

```
E ‖C_λ^{-1/2} (r − γ v_i)‖²  ≈  α · rank(C)          (≈ P−1 if C has full rank and λ → 0)
E ‖r − γ v_i‖²              =  α · tr(C) = α · mean‖v‖²           (that is `rel_err²` of §1.3)
```

Plain and whitened residuals checking out against these formulas validates the covariance `αC` *before* any decoding is done; if they do not, the decoder gaps in panel (a) cannot be trusted as evidence for or against whitening. A cheap further diagnostic is to compare the empirical covariance of `r − γ v_i` (over many probes) against `αC` along the leading eigenvectors of `C`.

### 2.11 Limits of the model

1. **Self-count.** Pairs sharing the target's value add extra noise along `u_t`; for the most frequent candidates this is not negligible (variance `(count_t−1)/d` on a unit signal, i.e. `≈ απ̂_t` relative).
2. **Key structure.** The noise scale `α = (P−1)/d` assumes isotropic keys. With `d_eff < d`, replace `d` by `d_eff` (this is what a `β_fit/β_th < 1` indicates).
3. **Gaussianity** relies on many independent `g_j` per candidate; for rare candidates with very few pairs it is exact only if the `g_j` are Gaussian, which holds for large `d`.
4. **In-sample evaluation.** Probes are sampled from the *stored* pairs and `π̂` includes the probe's own count, so the prior is slightly optimistic for rarely-stored values (a probe whose value was stored once sees `count = 1` rather than 0). A leave-one-out prior (subtract 1 from the true candidate's count) removes the bias.
5. **Constant-per-candidate noise.** The `log det` terms that would depend on the candidate are dropped; with strongly non-uniform `C` along the candidates this is an approximation.
6. **Hyper-parameters.** `ridge` and `prior_smoothing` are not fitted by the 2-fold scheme; choose them on a held-out set of probes.
