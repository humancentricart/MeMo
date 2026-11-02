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


Substituting aboce, we have:
```
d_eff ~ (P-1)power_ratio
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
