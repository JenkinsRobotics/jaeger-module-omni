# JaegerOmni 41M Causal Balanced

Development candidate blending 15% of the imported GemmaOmni final-window
causal trunk into 85% of JaegerOmni 41M Medium. The output keeps the explicit
causal mask and the imported frame-level perception heads.

The blend was selected on the predeclared dev-clean/AMI frontier, then run once
on the remaining corpus gates and cases12. It improves four of five corpus
gates and holds 12/12 QA, but adds two dev-clean edits. It therefore does not
replace the current 1:1 checkpoint yet.

The perception heads require revalidation: their weights came from the exact
imported model, while the trunk below them changed through interpolation.
Until head metrics are rerun, the node exposes their logits for diagnostics
but no live barge/endpoint policy may consume them.
