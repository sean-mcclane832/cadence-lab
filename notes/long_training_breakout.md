# Long training on One-Frame Breakout (cadence-net 0.89.0)

Run and code live in [cadence-arcade/classic](https://github.com/sean-mcclane832/cadence-arcade/tree/master/classic);
page: https://sean-mcclane832.github.io/cadence-arcade/breakout-long/. This note keeps the lab-side reading.

**Setup.** MinAtar Breakout, but players see one frame of hand features (ball column/row, paddle
column, their difference), so the ball's direction is only knowable from a moment ago. A coach labels
every training moment; Cadence learns by `Brain.lesson` then `act`. 300 blocks x 1,000 moments per
life (10x the first run), 12 seeds (10-21). Primary: bricks per test game over the last 10 blocks.

| player | 30 blocks | 300 blocks, median [IQR] |
| --- | --- | --- |
| Cadence, recall + position term | 5.65 | 13.15 [11.82-13.74] |
| Cadence, recall, `recall_position=0` | 5.57 | 10.38 [7.98-12.33] |
| Cadence, no recall | 6.97 | 13.35 [10.50-14.95] |
| plain MLP, one frame | 5.10 | 6.02 [5.84-6.18] |
| plain MLP, two frames (reference) | 6.72 | 19.20 [17.70-20.38] |

Paired by seed (Wilcoxon): Cadence (any arm) > one-frame MLP 12/12, p = 0.0005, about +7.
Two-frame MLP > position-term Cadence 12/12 (+6.2), > no-recall 11/12 (+5.9). Position term vs
`recall_position=0`: 7/12, +2.97, p = 0.09.

## Reading

- **The settle is a working memory.** A memoryless learner with one frame stops near 6 bricks; it
  cannot know the ball's direction. Every Cadence arm, including the one with recall switched off,
  climbs past 13. The relaxed state carries the previous moment forward, and that alone is worth
  about 7 bricks here.
- **But it is a lossy one.** Handing a plain MLP the second frame explicitly beats every Cadence
  arm by about 6 after long training. What the settle carries is real but noisier than the frame
  itself. At 30 blocks this gap did not exist; the short run hid it.
- **Recall memory adds nothing on top of the settle in this task**, with or without the 0.89.0
  position term. The position term's lean over plain recall (+3, 7/12) is worth a seed-doubled rerun
  before anyone calls it either way.
- **Short runs mislead in both directions.** At 30 blocks the two-frame reference and Cadence were
  level, and the position term was a clean null. Both readings moved with 10x training. Learning
  curves should be checked for a plateau before ranking players.

## Next

- 24 more seeds of pos32 vs pos0 at 300 blocks, to settle the position-term lean.
- Probe what the settle carries: decode last-moment ball direction from the settled state.
