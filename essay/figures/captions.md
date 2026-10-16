# Figure captions and placement

Use these captions in the manuscript rather than placing long interpretation in the graphics.

**Fig. 1.** Matched global-surrogate comparison at nominal per-target B=27. The upper panel gives all-seed targetwise mean bounded loss; the lower panel gives all-case, targetwise-trimmed, and worst-seed summaries. The four variants form a 2×2 design: GP or ten-network ensemble, each using either 17 seeded random model-building samples or six initial samples followed by 11 balanced L/R uncertainty acquisitions. All variants model both responses, use α=0.5 when active, and apply the same ten-point resistance-aware final ranking and decision rule.

**Fig. 2.** Direct-optimizer comparison across the 87 targets. The upper panel gives trimmed bounded resistance loss; the lower panel gives the trimmed number of physical-model evaluations. Each point is the mean after omitting the largest and smallest of ten independent runs. Curves use shape-preserving cubic Hermite interpolation solely to improve readability; symbols retain the computed values.

**Fig. 3.** Global dual-response GP uncertainty sampling versus simulated annealing. The upper panel gives trimmed bounded resistance loss; the lower panel gives physical-model evaluation count. Markers show every integer target, including all 87 observations on the GP's constant B=27 trace. GP counts 17 shared model-building evaluations (six initial and 11 active acquisitions) and ten target-specific final evaluations; SA is restarted independently for every target. Both methods use the same ±2.5% relative-inductance criterion and omit the extreme two of ten runs at each target.

**Fig. 4.** Matched nominal per-target B=27 comparison of the reusable balanced dual-response global GP with pure target-conditioned GP-cEI, probability-weighted GP-cLCB, and weighted GP-IVR. The upper panel gives all-seed mean bounded loss at each target; the lower panel gives all-case, targetwise-trimmed, and worst-seed summaries on a logarithmic scale. Every target-conditioned method models both L and R and uses ten seeds, six initial samples, 11 acquisitions, and ten final evaluations per standalone target. cEI gives the best exact recovery and loss summaries. cLCB has two more exact recoveries than IVR, while IVR has substantially lower loss severity.

**Fig. 5.** Fresh-seed nested S3 budget audit. Every point uses
$B=6+x+12$, $x=6,\ldots,12$. Pure cEI is compared with the frozen
cEI/GP-global/disagreement/NTK mixture $(.55,.15,.15,.15)$. The policies use
identical initial designs, the same oracle rows within each trajectory, and a
common GP-only 12-point confirmation rule. The upper panel gives mean bounded
loss; the lower panel gives exact-optimum recovery.

**Fig. 6.** Nominal-alignment 8-D WPT results at $B=6+32+12=50$. The left
panel gives the fraction of seeds returning $M=1500$ nH within $\pm2.5\%$;
the right panel gives mean and standard deviation of resistance among
qualified runs. The fifth bar is the lower-exploration
$(.85,.05,.05,.05)$ control. Qualified-only resistance must be interpreted
jointly with qualification. Pose is fixed, and misalignment or fabrication
tolerance is not optimized.
