# research-relay-lab

A small, fully checkable study of **where a deep-research pipeline loses the evidence
it found**. One frozen 142k-parameter policy is wired into six harnesses that differ
only in what is allowed to travel from the researcher to the reporter, and the same
rollouts are then decomposed into the three failures an end-to-end score blurs
together: the researcher walked to the wrong record, it read the right record and
wrote the wrong value, or it wrote the right value and the notebook never carried it.

The motivation is [bytedance/deer-flow](https://github.com/bytedance/deer-flow), a
deep-research harness whose pipeline is Planner → Research Team → **Reporter that
aggregates the team's findings** — the same shape as every planner/researcher/reporter
agent now being shipped. Its published evaluations measure the report. This repo
measures the relay between the findings and the report, on a task with an exact
verifier, and reports what a harness does not usually admit:

- a reporter handed **raw search hits** can be near chance while losing *no facts at
  all* — every line it needs is in the window, and the evidence still does not become
  an answer;
- one slot short of the chain, **every report cites more hops than its own notebook
  holds**, and many of them are the exact arithmetic fold of the truncated chain:
  consistent with the notes kept, wrong about the question asked;
- a pipeline that plans the whole route from one search cannot be rescued by any
  notebook size, because the facts were never gathered.

CPU-only. No API keys, no dataset downloads, no pretrained weights, no external model.

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
pytest -q
```

## Quickstart

Fold one query with the exact verifier — no model involved, so a reader can check the
ground truth without trusting a network, and see what a search has to return for a
record whose freshest line is buried among stale ones and other records entirely:

```
$ relay ledger --seed 5 --depth 3
```

```
query: r19 d3   ledger seed 5: 134 lines over 64 records
hop 1: r19 has 2 stale line(s); a search returns 3 lines, freshest = r19 v1 -> r11
       write v1, walk to r11, running sum v1
hop 2: r11 has 0 stale line(s); a search returns 3 lines, freshest = r11 v2 -> r42
       write v2, walk to r42, running sum v3
hop 3: r42 has 2 stale line(s); a search returns 3 lines, freshest = r42 v1 -> r25
       write v1, walk to r25, running sum v4
answer: v4 (v1 + v2 + v1 mod 5)
```

Watch a token budget take a fact away, with a flawless researcher supplying the notes,
so `lost` and `overclaim` mean something a reader can compute by hand before they mean
something in a table:

```
$ relay notebook --arm cited_relay --budget 4 --depth 3
```

```
arm cited_relay: 4 tokens of notebook at 2 per fact, so 2 of 3 hops fit
report payload: v1 r19 v2 r11
head 4 + payload 4 = 8 tokens the reporter reads; carried 2, lost 1
a report that cites 3 hops cites more than the interface delivered; folding the 2 surviving notes gives v3 against a true v4
```

Train the shared policy on a small split and print the six harnesses side by side. This
is two minutes and fifty-five seconds on the machine this was written on (CPU, no GPU
involved); the tables below come from the full three-seed run instead, so the digits
here are not the digits there:

```
$ relay demo
```

```
seed 0: 2500 train / 120 held-out queries, 1200 steps, 141,990 params, CPU
shared policy          teacher-forced  note 0.977  next 0.933
typed_relay            B=3   acc 0.858  walked 0.917  misread 0.025  lost 0.000  over 0.000  tok 7.0
cited_relay            B=6   acc 0.850  walked 0.917  misread 0.025  lost 0.000  over 0.000  tok 10.0
full_context           B=36  acc 0.175  walked 0.917  misread   n/a  lost 0.000  over 0.000  tok 34.0
plan_first             B=3   acc 0.192  walked 0.467  misread 0.015  lost 0.583  over 1.000  tok 7.0
single_shot            B=0   acc 0.217  walked   n/a  misread   n/a  lost 1.000  over 1.000  tok 4.0
oracle_relay           B=3   acc 0.975  walked 0.917  misread 0.025  lost 0.000  over 0.000  tok 7.0
typed_relay frontier         B=1:0.208  B=2:0.508  B=3:0.858  B=4:0.858
cited_relay frontier         B=2:0.208  B=4:0.542  B=6:0.850  B=8:0.825
full_context frontier        B=4:0.217  B=12:0.192  B=24:0.175  B=36:0.175
```

`n/a` is not a low number: `single_shot` never searches, so it has no walk to be right
about, and `full_context` never writes a note, so it cannot be caught misreading one.

Reproduce the whole study and re-render the block below:

```
$ python experiments/run_study.py
$ python experiments/make_report.py --write
```

## Design

**The task.** A ledger of 64 records, each with one to three versions of a line
`r07 v4 r12`: standing on `r07`, its current value is `v4`, and the chain continues at
`r12`. A query names a record and a depth of two or three hops; the answer is the sum,
modulo five, of the values on the records reached by always taking the freshest line.
`relay.kb.truth` computes it with no model in the loop, which is the entire reason this
repo can make the claims it makes.

Every query gets its **own** ledger. That is not decoration: with one shared ledger a
policy can memorise "record 7 is worth 4" and answer without reading the evidence it
was handed, and the study would then measure memorisation instead of the relay. The
rule repeats; the facts never do.

A search for a record returns its freshest line plus two lines from unrelated records,
in a shuffled order (`relay.harness.look`). So "was the evidence available" and "was
the evidence used" are different questions, and the harnesses differ on both.

**The policy.** A 72-wide, 4-head, 3-layer transformer encoder over a token sequence,
sum-pooled into four typed heads — `note` (what value to write here), `nxt` (which
record to search next), `ans` (the reported answer), `cite` (how many hops the report
claims). Behaviour cloning on oracle traces, both roles trained from the same rows, and
the reporter is trained on **both** payload shapes the arms will hand it — a notebook
and a raw hit list — so no arm is compared while untrained. The readout is a sum rather
than a mean because the reporter's job is modular addition over a variable number of
notes: a mean would let the *count* of notes leak into the magnitude of each one, and
padding would silently rescale the answer.

**The harnesses.** Six arms around that one frozen policy. A researcher role searches,
writes a note, and moves; a reporter role folds what it is given. What differs is the
interface between them, priced in one shared currency — tokens the reporter has to
read:

| what travels | tokens per fact |
|---|---:|
| a typed note (`typed_relay`, `plan_first`, `oracle_relay`) | 1 |
| the same note plus the record it came from (`cited_relay`) | 2 |
| a raw search hit, three lines (`full_context`) | 4 |
| nothing (`single_shot`) | 0 |

`oracle_relay` is the control: the notebook is written by the verifier, so whatever it
gets wrong it does not get wrong by reading. `plan_first` searches once and then plans,
which is what a pipeline does when it decides the whole route up front. `single_shot`
answers before searching, and is the reason the other rows can be called
evidence-driven.

**The decomposition.** `walked_ok` (did the chain of records match),
`wrong_value_rate` (of the hops it took, how many did it write the stale or wrong
value), `lost_rate` (facts that never reached the reporter), `carried_rate`,
`cite_overclaim` (the report cites more hops than the notebook holds),
`partial_sum_rate` (the answer is the exact fold of a truncated chain),
`acc_full_notebook` (the ceiling, on the queries where nothing was lost), and a
per-hop misread profile. A condition an arm cannot be observed under is reported as
`-`, never as 0.0.

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/frontier.json`](results/frontier.json); the tables and the sentences around them are rendered by `experiments/make_report.py`. 3 seeds (0, 1, 2), 4,000 training and 1,200 held-out queries per seed, 2,500 steps at lr 0.005, and 141,990 parameters shared by every harness.*
*Measured under Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu. A rerun inside that environment reproduces the artifact field for field except the wall-clock; elsewhere the thread count changes the float reduction order and the last digits move.*

### The six harnesses, at the budget each would choose

| harness | what travels from researcher to reporter | tokens per fact | hops searched | reporter tokens | answer acc | walked the chain | misread a hop | lost in relay | cites more than it holds |
|-----|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| `oracle_relay` | a typed note per hop, written by the verifier | 1 | the whole chain | 7.0 | 0.993 +/- 0.003 | 0.980 | 0.010 | 0.000 | 0.000 |
| `typed_relay` | a typed note per hop | 1 | the whole chain | 7.0 | 0.955 +/- 0.015 | 0.980 | 0.010 | 0.000 | 0.000 |
| `cited_relay` | a typed note plus the record it came from | 2 | the whole chain | 10.0 | 0.945 +/- 0.013 | 0.980 | 0.010 | 0.000 | 0.000 |
| `full_context` | the raw search hits | 4 | the whole chain | 34.0 | 0.291 +/- 0.026 | 0.980 | - | 0.000 | 0.000 |
| `plan_first` | one search, then a plan the reporter has to trust | 1 | one hop, then it guesses the rest | 7.0 | 0.203 +/- 0.013 | 0.495 | 0.002 | 0.583 | 1.000 |
| `single_shot` | nothing: the reporter answers before any search | 0 | no hop | 4.0 | 0.203 +/- 0.001 | - | - | 1.000 | 1.000 |

*A `-` is not a zero: it means the harness cannot be asked that question. `single_shot` never walks, so it has no walk to be right about; `full_context` hands over raw hits instead of writing a note, so there is no note of its own to misread and no truncated notebook to fold. Chance on five possible answers is 0.200.*

**The notebook is not the bottleneck at this budget, and the reader is not either.** `oracle_relay` is the control: the same harness, the same window, with the notes written by the verifier instead of the researcher. It answers **0.993**, the arm that reads its own evidence 0.955 (+0.038, up on 3 of 3 seeds), and neither loses a fact: `lost_rate` is 0.000 for both. What separates them is 0.010 of hops written with the wrong value -- the researcher's share of the error, billed to the stage that reads rather than the stage that carries.

The reading itself is nearly solved: given the true chain and its evidence, the frozen policy writes the right value 0.989 of the time and picks the right next record 0.981 of the time (`relay.study.teacher_forced`, on the same held-out queries). Everything the arms lose on top of that belongs to the relay or to the walk the policy takes for itself.

**Provenance buys tokens, not accuracy.** `cited_relay` carries the record id beside every value (2 tokens per fact instead of 1), so its default budget holds the same three hops while its reporter reads 1.4x the tokens (10.0 against 7.0). At that equal capacity it scores 0.945 against `typed_relay`'s 0.955: -0.010, down on 3 of 3 seeds. That difference is inside the across-seed spread (0.015), so this study does not claim provenance *costs* accuracy -- it claims provenance is paid for in context and returns nothing measurable here.

**Evidence in the window is not evidence the reporter can use.** At its default budget `full_context` loses nothing at all: every hop's freshest line fits, so `lost_rate` is 0.000 and `carried_rate` 1.000 -- and it still answers **0.291**, against `typed_relay`'s 0.955 on 7.0 reporter tokens instead of 34.0. The failure is *when* the selection happens. A note is written after the researcher has already resolved which line of the record is current; a raw hit list hands the reporter the current line plus 2 distractors per hop and asks it to resolve that at read time. On three-hop questions, where all 9 lines are present, the arm sits at 0.202 against a chance level of 0.200. `dump everything into the context` is not a free substitute for writing things down.

### The same harnesses as the notebook window is cut

| notebook budget (tokens) | `typed_relay` | `cited_relay` | `full_context` | `plan_first` |
|-----|------:|------:|------:|------:|
| 1 | 0.198 | - | - | 0.198 |
| 2 | 0.593 | 0.198 | - | 0.196 |
| 3 | 0.955 | - | - | 0.203 |
| 4 | 0.956 | 0.596 | 0.194 | 0.201 |
| 6 | - | 0.945 | - | - |
| 8 | - | 0.950 | - | - |
| 12 | - | - | 0.196 | - |
| 24 | - | - | 0.287 | - |
| 36 | - | - | 0.291 | - |

The same grid, as the fraction of the chain that never reaches the reporter:

| notebook budget (tokens) | `typed_relay` | `cited_relay` | `full_context` | `plan_first` |
|-----|------:|------:|------:|------:|
| 1 | 0.583 | - | - | 0.583 |
| 2 | 0.167 | 0.583 | - | 0.583 |
| 3 | 0.000 | - | - | 0.583 |
| 4 | 0.000 | 0.167 | 0.864 | 0.583 |
| 6 | - | 0.000 | - | - |
| 8 | - | 0.000 | - | - |
| 12 | - | - | 0.559 | - |
| 24 | - | - | 0.153 | - |
| 36 | - | - | 0.000 | - |

A budget is not a number of facts. At four tokens the window holds 4 notes for `typed_relay`, 2 for `cited_relay`, and 1 raw line for `full_context` -- and accuracy is 0.956 for `typed_relay`, 0.596 for `cited_relay`, 0.194 for `full_context`, 0.201 for `plan_first`. Same tokens, same frozen policy: the interface set the capacity.

**One slot short and the report stops believing itself.** Cut `typed_relay` to one token: accuracy 0.955 -> 0.198, 0.583 of the chain never arrives, and **1.000 of reports cite more hops than their own notebook holds.** A notebook that lost a fact is not a notebook that says so: the citation head was trained on full notebooks and keeps claiming the hop count the query asked for. The reporter cannot notice the loss from the inside -- only the interface can.

What it does instead is add up what is left. At two tokens, where the walk is still right 0.980 of the time, 0.133 of answers are the exact fold of a truncated chain and 0.500 cite more than they hold. That is the failure shape a deep-research pipeline shows a user: a summary that is arithmetically consistent with the notes it kept, reads as though it were built from every source it cites, and is wrong about the question that was asked.

`plan_first` is the arm that decides the whole route from one search, and no budget can rescue it: accuracy stays between 0.196 and 0.203 across its whole sweep with a window that carries 0.417 of the chain and a walk that is right 0.495 of the time. The facts were never gathered, so widening the notebook only gives the reporter more room to guess in.

### The same budget, by how deep the chain is

| harness | two-hop acc | three-hop acc | two-hop lost | three-hop lost |
|-----|------:|------:|------:|------:|
| `oracle_relay` | 1.000 | 0.986 | 0.000 | 0.000 |
| `typed_relay` | 0.982 | 0.928 | 0.000 | 0.000 |
| `cited_relay` | 0.982 | 0.908 | 0.000 | 0.000 |
| `full_context` | 0.379 | 0.202 | 0.000 | 0.000 |
| `plan_first` | 0.201 | 0.205 | 0.500 | 0.667 |
| `single_shot` | 0.203 | 0.204 | 1.000 | 1.000 |

Every extra hop is one more fact that has to survive, and one more chance to misread it. Going from two to three hops costs the verifier-written notebook -0.014 and the researcher-written one -0.054, at a budget where neither loses a fact (`lost` 0.000 at the bottom). The depth penalty is mostly a reading penalty, not a capacity one -- which is exactly the part an end-to-end score folds into a single number.

Misreads by position in the chain, over the same `typed_relay` rollouts: hop 1 0.004, hop 2 0.010, hop 3 0.024. The rate rises with distance from the record the query named: the same policy reads 0.989 of a *true* chain's hops right when the harness hands it the record to stand on, and misreads 0.010 of the hops on the walk it takes for itself. No end-to-end score reports this.

### Per seed, so a gap carried by one initialisation stays visible

| seed | `oracle_relay` | `typed_relay` | `cited_relay` | `full_context` | `plan_first` | `single_shot` |
|-----|------:|------:|------:|------:|------:|------:|
| 0 | 0.988 | 0.933 | 0.927 | 0.270 | 0.202 | 0.203 |
| 1 | 0.995 | 0.963 | 0.951 | 0.328 | 0.188 | 0.203 |
| 2 | 0.995 | 0.968 | 0.958 | 0.275 | 0.220 | 0.204 |

Read every gap above against the widest across-seed standard deviation in these tables, 0.026 (the steadiest arm is `single_shot` at 0.001); the provenance comparison is the one this report declines to claim precisely because it sits inside it. `single_shot` answers at 0.203 against a chance level of 0.200: with no evidence the reporter is guessing, which is what makes the other rows evidence-driven rather than prior-driven.

<!-- RESULTS:END -->

## What this does not show

- **It is arithmetic, not research.** No language, no ranking, no retrieval quality,
  no long documents. The claim is about the *shape* of the failure — where in a
  researcher/reporter split evidence stops being usable, and that a report's own
  citation claim does not move when the notebook loses a fact. Nothing here says a real
  reporter fails at the rate measured.
- **The raw-hit result is a readout result.** `full_context` loses every comparison
  here with a sum-pooled, class-emitting reader that has to select the current line out
  of three at read time. A decoder that reads its context token by token may well do
  that selection; this repo does not test that, and the raw-hit row should not be read
  as a general argument against long context.
- **The capacity cliff is a design fact, not a discovery.** At budget 1 the harness
  keeps the first `capacity` notes because that is what a fixed window does. What is
  measured is what the reporter and the citation head *do* about it, which was not
  known beforehand and is not identical at every seed.
- **One policy family, one training recipe.** The comparison is between harnesses at a
  fixed policy, so an arm's number may not survive a stronger policy — and the
  across-seed spread printed above is a spread over initialisations and ledgers, not a
  confidence interval over architectures. The provenance gap is inside that spread and
  is labelled as such rather than claimed.
- **Two depths.** `d2` and `d3` are enough to put a budget on both sides of the chain
  length, and nothing more. At these depths the verifier-written notebook is close
  enough to perfect that the fold is not the binding constraint; a deeper chain would
  eventually make the arithmetic itself the thing under test, which would confound the
  relay.
- **No reranker, no index, no tool use.** `look()` resolves revisions and shuffles hits
  with a fixed number of distractors, so the evidence set is a constant rather than a
  variable and none of the reported differences can be attributed to retrieval.

## Reproducing

```
python experiments/run_study.py --out results/again-check.json --against results/frontier.json
```

re-runs all three seeds and fails if any field of the fresh artifact differs from the
committed one — `runtime_sec` is the only field a rerun is allowed to move. In the
recorded environment the two files are identical field for field; across machines the
thread count changes the float reduction order and the last digits can move, which is
why the artifact carries its own `environment` block and every table carries its
spread. `python experiments/make_report.py --check` fails if the README block has
drifted from the artifact, and the test suite recomputes the artifact's means from its
own per-seed numbers, so a table cannot be edited into agreement with a story.

## Layout

```
src/relay/kb.py        the ledger, the chains, and the exact verifier
src/relay/model.py     the shared 72-wide transformer and its four heads
src/relay/harness.py   the six arms: what travels, and what a budget buys
src/relay/train.py     behaviour cloning on oracle traces, both roles
src/relay/metrics.py   the decomposition, and what an arm cannot be asked
src/relay/study.py     three seeds, one budget sweep, one aggregate
src/relay/cli.py       ledger / notebook / demo
experiments/run_study.py       writes results/frontier.json
experiments/make_report.py     renders the README block from that artifact
tests/                 the guards above, plus byte-pinned transcripts
```

## License

MIT. See [LICENSE](LICENSE).
