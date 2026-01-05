# Relational State Estimation from MoE Router Logits

*Draft for Hidden State Substack*

## TL;DR

Router logits (64-dim) retain 94-99% of residual stream (2048-dim) signal for intent, emotion, and tension classification—32× compression with minimal information loss. Power encoding is more nuanced: weak signal on Wikipedia Talk (speaker identity), pending results on Enron (power-in-action).

## Background

Prior work established that 64-dimensional MoE router logits encode formality at parity with 2048-dimensional residual stream activations — a 32× compression with minimal information loss. But formality is a mono-axial signal: a property of individual text samples.

This work asks: **do router logits also encode relational signals?** Properties of speaker relationships rather than individual text:
- Power differential (who has higher status)
- Emotional valence (happy vs sad vs angry)
- Intent categories (asking vs telling vs requesting)
- Tension dynamics (escalating vs repairing)

## Results

| Signal | Router AUC | Residual AUC | Retention |
|--------|-----------|--------------|-----------|
| Intent (4-class) | 0.841 | 0.877 | 96% |
| Emotion (7-class) | 0.879 | 0.938 | 94% |
| Power/Wikipedia | 0.608 | 0.677 | 90% (inconclusive) |
| Power/Enron | TBD | TBD | TBD |
| **Tension** | **0.995** | **1.000** | **99.5%** |

**Key finding:** Router logits strongly encode content-type signals (intent, emotion, tension dynamics). The power result requires careful interpretation—see below.

## The Power Probe Problem

Our initial power probe on Wikipedia Talk Pages yielded weak signal (AUC 0.608), which we initially interpreted as: "routers are content-typed, not speaker-typed."

**But this conclusion was too strong.** Reviewer feedback (credit: Herbie) correctly identified that Wikipedia Talk tests speaker *identity* (admin label), not power *being exercised*:

1. **Admin status is a noisy proxy.** An admin asking "what do you think?" looks identical to a non-admin asking the same question. The label tells us *who* is speaking, not *how* they're speaking.

2. **Power manifests linguistically.** When a senior exercises authority, they use directives, delegation language, and expect compliance. When a junior addresses a senior, they hedge, defer, and request rather than command.

3. **The interesting question:** Do routers distinguish "directive from senior" vs "directive from junior"? This requires data where the direction of power is clear.

**Enter Enron.** The Enron email corpus includes sender/recipient metadata that can be mapped to corporate seniority. We can label emails as:
- **Downward** (senior→junior): CEO to manager, VP to analyst
- **Upward** (junior→senior): Analyst to VP, manager to CEO

This tests whether routers encode power *when exercised* rather than speaker metadata. If downward/upward classification succeeds (AUC ≥0.70), routers see power-in-action. If it fails similarly to Wikipedia (~0.60), routers truly don't encode relational power dynamics.

**Status:** Enron experiments pending. Results will update this section.

## Implications

Router logits encode relational signals with high fidelity:

1. **Real-time social state estimation** — Monitor social dynamics during conversations at ~32× less compute than full activation analysis.

2. **Features for S3AP parsing** — The CMU Social World Models work (Zhou et al., 2025) shows that explicitly parsing social state (mental states, goals, relationships) dramatically improves ToM reasoning. Router probes could provide cheap input features for this parsing.

3. **Tension maintenance detection** — In interactive fiction, AI often collapses into compliance when it should maintain narrative tension. A router probe could detect this collapse in real-time (AUC 0.995!).

4. **Power detection (if Enron confirms)** — Could enable real-time monitoring of hierarchical dynamics in organizational communications.

## Methods

**Model:** OLMoE-1B-7B (64 experts, fully open)

**Extraction:** Forward pass with hooks on router gates, mean pooling over tokens

**Probing:** Logistic regression on 64-dim router logits vs 2048-dim residual stream

**Datasets:**
- DailyDialog (intent, emotion)
- Wikipedia Talk Pages (speaker identity)
- Enron emails (communication direction)
- Synthetic tension pairs (Claude-generated)

## Conclusion

MoE routers encode more than we thought. The 64-dimensional router logit space captures intent, emotion, and tension dynamics at near-parity with the full residual stream. Whether they also encode relational power depends on how we operationalize "power"—as speaker identity or as linguistic exercise. Enron experiments will tell.

## References

- Muennighoff et al. (2024). OLMoE.
- Zhou et al. (2025). Social World Models.
- Zhou et al. (2024). SOTOPIA.
- Lai et al. (2025). SAFEx.
- Klimt & Yang (2004). The Enron Corpus.
