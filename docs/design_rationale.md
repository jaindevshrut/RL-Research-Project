# Design Rationale (decision log)

A short journal of the choices that shaped this codebase, so a reviewer
can ask "why did you pick X over Y?" and get a one-paragraph answer.

## D1 — Why DQN instead of A2C/PPO (the paper's choice)?

The paper's appraisal extractor reads off `motivational_relevance`,
`novelty`, and `accountable`. Notably absent: a clean, on-policy
**Q-value spread** and an explicit **V(s)**. A2C *does* train a value
function, but it tracks V(s), not per-action Q — so "power" (the spread
of value across actions) requires a workaround.

DQN gives `Q(s,·)` natively, and Dueling DQN gives `V(s)` as a named
output. So the dim "power" gets exactly the signal Scherer talks about
(action-effect range), and "anticipation" gets the absolute V level.

Risk: DQN is less sample-efficient on continuous-control problems; here
the action space is discrete (5 actions) so DQN is the natural fit.

## D2 — Why a bootstrap ensemble (K=3)?

We need an **epistemic** uncertainty signal for the predictability
appraisal. Options:
* MC dropout — cheap, but conflates structural/aleatoric noise.
* Bayesian last-layer — adds inference overhead.
* Bootstrap ensemble — K independently-initialised heads, each trained
  with sub-sampled experience. Disagreement is direct, interpretable,
  and well-studied (Osband et al., 2016).

K=3 is small enough to barely cost more than vanilla DQN (only the
heads are duplicated; the trunk is shared) and large enough that the
std across heads is informative. K=5 is ideal but adds overhead.

## D3 — Why these four NEW dims and not, e.g., "moral norm conformity"?

We used three filters:

1. *Computable from RL signals already present in the agent.*
   Norm-conformity would require an external norm model — out of scope.
2. *Statistically distinct from the existing four.*
   We dropped candidates that decompose into the same TD-error signal
   (e.g. "discrepancy" reduces to |TD|).
3. *Mappable onto an emotion the paper's 4-dim vector cannot
   currently distinguish.*
   - **Predictability** distinguishes anxiety from determined anger.
   - **Anticipation** distinguishes hope (high V, low realised reward yet)
     from current joy.
   - **Urgency** distinguishes panic from despair.
   - **Familiarity** distinguishes boredom from neutral.

## D4 — Why a custom GridWorld and not MiniGrid?

Three reasons:
* Zero install pain on Windows (the original repo's MiniGrid pin is
  outdated and doesn't `pip install` cleanly on modern setups).
* Counts of states are tiny (~ grid_size² × {has_key, ¬has_key}), which
  makes count-based novelty / familiarity exact instead of approximated.
* Every transition is hand-traceable in a debugger, which makes the
  appraisal numbers explainable line-by-line.

If the professor pushes back ("why not the same env as the paper?"),
the project structure makes the env easy to swap: replace
`src/env/gridworld.py` with a Gym/MiniGrid wrapper that exposes
`obs`, `step()`, `state_id()`. Nothing else changes.

## D5 — Why prioritised replay?

Two payoffs:

1. *Faster learning.* Standard PER benefit.
2. *Cleaner appraisal signals.* PER weights the gradient toward
   high-|TD| transitions — exactly the events that the appraisal
   vector most needs to characterise. This couples the training signal
   with the appraisal extractor's "interesting moments" prior.

The PER implementation is naive (no SumTree). For our buffer size
(50k) it is fast enough; for >10⁵ a SumTree is recommended.

## D6 — Why post-hoc decorrelation, not in-loop?

We considered orthogonalising the appraisal vector inside the agent
loop (Gram-Schmidt at each step). We decided against it:

* It would mix the appraisal signals together, breaking the one-to-one
  correspondence with the named CPM checks.
* The decorrelation analysis in `analysis/correlation_analysis.py` is
  *evidence* that orthogonality is achieved by *construction*, not
  enforced after the fact. That's the stronger scientific claim.

If the eventual downstream task (e.g. regression onto human emotion
ratings) wants a decorrelated input, run Gram-Schmidt as a
preprocessing step on the saved `appraisals.npz`. The shapes line up
for that.
