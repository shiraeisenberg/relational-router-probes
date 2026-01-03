# Relational State Estimation from MoE Router Logits

*Draft for Hidden State Substack*

## TL;DR

Router logits encode [RESULTS TO BE ADDED].

## Background

Prior work established that 64-dimensional MoE router logits encode formality at parity with 2048-dimensional residual stream activations — a 32× compression with minimal information loss. But formality is a mono-axial signal: a property of individual text samples.

This work asks: **do router logits also encode relational signals?** Properties of speaker relationships rather than individual text:
- Power differential (who has higher status)
- Emotional valence (happy vs sad vs angry)
- Intent categories (asking vs telling vs requesting)
- Tension dynamics (escalating vs repairing)

## Results

[TO BE COMPLETED AFTER EXPERIMENTS]

## Implications

If router logits encode relational signals:

1. **Real-time social state estimation** — Monitor social dynamics during conversations at ~32× less compute than full activation analysis.

2. **Features for S3AP parsing** — The CMU Social World Models work (Zhou et al., 2025) shows that explicitly parsing social state (mental states, goals, relationships) dramatically improves ToM reasoning. Router probes could provide cheap input features for this parsing.

3. **Tension maintenance detection** — In interactive fiction, AI often collapses into compliance when it should maintain narrative tension. A router probe could detect this collapse in real-time.

## Methods

[TO BE COMPLETED]

## References

- Muennighoff et al. (2024). OLMoE.
- Zhou et al. (2025). Social World Models.
- Zhou et al. (2024). SOTOPIA.
- Lai et al. (2025). SAFEx.
