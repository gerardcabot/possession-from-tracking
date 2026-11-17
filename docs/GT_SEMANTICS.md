# Public ground truth: qualitative semantics study

One-time study allowed by PLAN §5.3(a). Done on 2026-10-25.

**Scope.** The public window only (video frames 11,185–18,684, five minutes of the first half). We
listed the GT label runs in that window and, around each of the 20 label changes and each of the
four long ball gaps, looked at aggregate tracking context: whether the ATD ball was observed, its
distance to the nearest line, its speed, the nearest home and away player distances, and the mean
player speed. Nothing outside the window was compared with the GT.

**Rule.** The conclusions below are written in words. No number in `configs/` or `src/` was fitted
to, tuned on, or selected with the public GT. Where a conclusion mentions a rough size ("about half
a second"), it only describes what was seen; the code uses values from `docs/PARAMETERS.md`, which
are argued from football and physics.

## Conclusions

1. **Ball over a line.** When the ball leaves the pitch in play, `D` starts when the ball is over the
   line, give or take a couple of frames. It does not wait for the players to stop. Back-dating `D`
   to the line crossing is the right convention.
2. **Whistle stoppages.** When the ball stays on the pitch (fouls and similar), `D` starts while the
   players are still decelerating: shortly after the mean player speed starts falling from its live
   level, well before the team is standing still. In one case the ball was already missing from the
   tracking when `D` started. The start of the collective deceleration is a reasonable anchor.
3. **Restarts.** `D` ends at the restart touch, never before it. The new spell starts with the ball
   already moving away from the taker; the label change sits from roughly the kick to about a second
   after it (later for a short goal kick passed inside the box). "The ball is in play when it is
   kicked and clearly moves" (Laws of the Game) describes it well. The new spell belongs to the team
   whose player takes the restart (free kick, throw-in, corner and goal kick cases all seen).
4. **Switches between teams.** `H↔A` changes happen around the new team's first touch, not after a
   sustained period of control: the label flips when the receiving player arrives at the ball, and
   sometimes while a pass is still travelling toward that player. The GT does not wait for "a few
   seconds" of control. Contested balls with both teams close by can produce very short spells of a
   second or less (the shortest seen were under one second), so the GT is not smoothed. The ATD
   nearest player is often not the GT owner at those moments, so the raw nearest-player signal is
   noisy even where the GT is unambiguous.
5. **Ball gaps.** The public window has few gaps (the ball is visible about 88% of the time there,
   versus 59% on the whole graded match). The two long gaps during live play kept the same team from
   entry to exit. One gap started during play and turned into a stoppage shortly after the ball
   disappeared. One gap covered a whole stoppage and ended when the ball reappeared at the restart.
6. **Minimum spell length.** No minimum. Spells of well under one second and of a few seconds both
   occur next to long spells of 30–40 s.
7. **Pre-kickoff.** The first graded frame of the first half is already a possession of the kicking
   team: the graded block starts at the kickoff. The second half is `D` until its kickoff (FAQ), so
   the first frames of each half follow "dead until the kickoff touch".
8. **Dead-ball ownership.** None, as documented: every stoppage frame is `D`.

## How these conclusions shaped the rules

- Exit crossings are detected and back-dated to the first frame the ball is wholly over the line
  (conclusion 1). Stoppages without a crossing start at the deceleration onset (conclusion 2).
- Restarts are placed at the first frame the ball has clearly moved after a settled ball, and the
  restarting team is the taker's team (conclusion 3).
- Team switches are back-dated to the first frame of the opponent's control dwell (conclusion 4).
  We still require a short dwell, because the ATD is much noisier than the corrected tracking the GT
  comes from, and flicker costs spells.
- Inside long ball gaps the team is kept unless the team that controls the ball after the gap
  differs from the one before it (conclusion 5).
- Each half starts dead until its detected kickoff (conclusion 7).
