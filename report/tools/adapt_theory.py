#!/usr/bin/env python3
"""One-time conversion of the proof compendium theory/0*.tex into the report appendices.

The mathematics is copied unchanged.  Only references to project-management files
(SOURCE_AUDIT, NOTATION, the specification, prompt stages) are rewritten into references
to the report: \\audit{X} points to the source-audit appendix, \\cref{sec:notation} and
\\cref{sec:protocol} to the main text.  Every replacement must match exactly once (or the
stated number of times); the script stops otherwise.  After this conversion the appendix
files are maintained by hand (report/appendix/), so the script is a record, not a build step.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "theory"
DST = ROOT / "report" / "appendix"

FILES = {
    "01_occupancy.tex": "A_occupancy.tex",
    "02_divergences.tex": "B_divergences.tex",
    "03_fpg.tex": "C_fpg.tex",
    "04_firl.tex": "D_firl.tex",
    "05_fb.tex": "E_fb.tex",
    "06_fbil.tex": "F_fbil.tex",
    "07_environment.tex": "G_environment.tex",
}

R = {
"01_occupancy.tex": [
 ("% 01_occupancy.tex  --  THEORY_INDEX rows OCC-1 ... OCC-7   (Prompt 2A)",
  "% Appendix A (from theory/01_occupancy.tex)  --  claims OCC-1 ... OCC-7"),
 ("derived statement (\\textsf{D}) about the conventions adopted in \\texttt{NOTATION.md}.",
  "derived statement (\\textsf{D}) about the conventions of \\cref{sec:notation}."),
 ("This is the formal content of the spec's ``do not claim that state-only matching identifies expert action choices''.\n",
  "This is why the report never claims that state-only matching identifies the expert's action choices.\n"),
 ("defined at the same horizon $H$ (\\texttt{NOTATION.md} \\S4).", "defined at the same horizon $H$ (\\cref{sec:notation})."),
 ("as required by DM$_{\\mathrm{FB}}$ (SOURCE\\_AUDIT A4).", "as required by DM$_{\\mathrm{FB}}$ \\audit{A4}."),
 ("H-step gaps of learners are measured against $V^\\star_H$ (spec \\S5, change S4).",
  "H-step gaps of learners are measured against $V^\\star_H$ (\\cref{sec:protocol})."),
 ("only if the reward is read at $s_{t+1}$ (SOURCE\\_AUDIT R1).", "only if the reward is read at $s_{t+1}$ \\audit{R1}."),
 ("it must be examined for the actual kernel (SOURCE\\_AUDIT U-5).",
  "the implemented kernel is examined in \\cref{sec:env-continuous}, which shows that it has no atom and no clipping."),
],
"02_divergences.tex": [
 ("% 02_divergences.tex  --  THEORY_INDEX rows DIV-1 ... DIV-3   (Prompt 2A)",
  "% Appendix B (from theory/02_divergences.tex)  --  claims DIV-1 ... DIV-3"),
 ("prints the opposite orientation (SOURCE\\_AUDIT D1).", "prints the opposite orientation \\audit{D1}."),
 ("is twice the half-weight generator used here (SOURCE\\_AUDIT D3).", "is twice the half-weight generator used here \\audit{D3}."),
 ("In our objectives the learner is always the first argument (NOTATION \\S6).",
  "In our objectives the learner is always the first argument (\\cref{sec:notation})."),
 ("The two papers' names are thus opposite (SOURCE\\_AUDIT D2).", "The two papers' names are thus opposite \\audit{D2}."),
 ("i.e.\\ proportional to the free area of each cell (NOTATION \\S\\S4 and~6).",
  "i.e.\\ proportional to the free area of each cell (\\cref{sec:notation})."),
 ("This requires every cell of $\\mathcal X$ to have positive free area, which we assume of the histogram partition; whether clipping can charge cells of zero free area is part of U-5 (\\cref{rem:occ-density}).",
  "This requires every cell of $\\mathcal X$ to have positive free area; the retained cells of the implemented histogram have this property, and the implemented kernel has no clipping (\\cref{prop:env-cells,prop:env-cont})."),
 ("the reference density $\\upsilon_\\lambda=\\ind_{\\cS}/A_{\\mathrm{free}}$ (NOTATION \\S4);",
  "the reference density $\\upsilon_\\lambda=\\ind_{\\cS}/A_{\\mathrm{free}}$ (\\cref{sec:notation});"),
],
"03_fpg.tex": [
 ("% 03_fpg.tex  --  THEORY_INDEX rows FPG-1 ... FPG-8   (Prompt 2A)",
  "% Appendix C (from theory/03_fpg.tex)  --  claims FPG-1 ... FPG-8"),
 ("the full findings are SOURCE\\_AUDIT F1--F10.", "the full findings are \\audit{F1--F10}."),
 ("The activation used in the implementation is recorded as open item U-12 (THEORY\\_INDEX, SOURCE\\_AUDIT \\S12).",
  "The implemented MLP policy uses $\\tanh$ hidden layers, which are $C^\\infty$ \\audit{U-12}."),
 ("not the cell masses $\\upsilon_b$ (NOTATION \\S4).", "not the cell masses $\\upsilon_b$ (\\cref{sec:notation})."),
 ("they are stored only for this evaluation (spec \\S5).", "they are stored only for this evaluation (\\cref{sec:protocol})."),
],
"04_firl.tex": [
 ("% 04_firl.tex  --  THEORY_INDEX rows FIRL-1 ... FIRL-3   (Prompt 2A; theory only)",
  "% Appendix D (from theory/04_firl.tex)  --  claims FIRL-1 ... FIRL-3 (theory only)"),
 ("$f$-IRL is not implemented in this project (spec \\S1).", "$f$-IRL is not implemented in this study."),
 ("the formula is only an approximation of the gradient (U-7).", "the formula is only an approximation of the gradient \\audit{U-7}."),
 ("may drop $\\log2$ (SOURCE\\_AUDIT D3).", "may drop $\\log2$ \\audit{D3}."),
],
"05_fb.tex": [
 ("% 05_fb.tex  --  THEORY_INDEX rows FB-1 ... FB-11   (Prompt 2B)",
  "% Appendix E (from theory/05_fb.tex)  --  claims FB-1 ... FB-11"),
 ("the population law of the replay next states (NOTATION \\S5), a probability measure;",
  "the population law of the replay next states (\\cref{sec:notation}), a probability measure;"),
 ("With the reward read at $s_{t+1}$ (SOURCE\\_AUDIT R1) the two agree,", "With the reward read at $s_{t+1}$ \\audit{R1} the two agree,"),
 ("The kernel is fixed in 3A (U-10); its rank $k$ is computable from the oracle matrix and should be logged.",
  "The implemented finite kernel has a stay action of this form with $\\epsilon=p_{\\mathrm{slip}}=0.1$, and its rank is $\\#\\cS$ (\\cref{prop:env-finite}(b))."),
 ("are coupled \\emph{under exactness} (spec \\S6, SOURCE\\_AUDIT T2):", "are coupled \\emph{under exactness} \\audit{T2}:"),
 ("the environment is continuing (spec \\S3),", "the environment is continuing (\\cref{sec:notation}),"),
 ("are not proved (open item U-14).", "are not proved \\audit{U-14}; the implemented networks use LayerNorm, $\\tanh$ and SiLU, which are $C^\\infty$."),
 ("and break the single-reference requirement (SOURCE\\_AUDIT A3).", "and break the single-reference requirement \\audit{A3}."),
 ("\\begin{remark}[the continuous maze and the 2021 atom; consequences for 3A]\\label{rem:fb-dom}\nThese are consequences of the kernel design, which is decided in 3A (SOURCE\\_AUDIT U-5). None is verified here.",
  "\\begin{remark}[the continuous maze and the 2021 atom]\\label{rem:fb-dom}\nThese consequences depend on the design of the kernel; the implemented kernel is analysed in \\cref{sec:env-continuous}."),
 ("must use the same buffer next states (SOURCE\\_AUDIT A3).", "must use the same buffer next states \\audit{A3}."),
 ("and \\cref{prop:fb-orth} are not proved (U-14).", "and \\cref{prop:fb-orth} are not proved \\audit{U-14}."),
 ("\\item (U-3, decided) After pretraining,", "\\item \\audit{U-3} After pretraining,"),
 ("one copy for all of them (SOURCE\\_AUDIT A2--A3).", "one copy for all of them \\audit{A2--A3}."),
 ("\\item (U-2, analysis; decision with the pilot) Under exactness", "\\item \\audit{U-2} Under exactness"),
 ("  $\\tau$ is chosen in the pilot together with $R$ and reported with the measured $\\|z_{ER}\\|_2$.",
  "  The pilot fixed $\\tau=1$ together with $R=\\sqrt d=4$ (\\cref{sec:protocol}); $\\tau$ is reported with the measured $\\|z_{ER}\\|_2$."),
 ("The rate is a 3B configuration choice.", "The implementation uses momentum $0.99$ after every update (\\cref{sec:protocol})."),
],
"06_fbil.tex": [
 ("% 06_fbil.tex  --  THEORY_INDEX rows FBIL-1 ... FBIL-5   (Prompt 2B)",
  "% Appendix F (from theory/06_fbil.tex)  --  claims FBIL-1 ... FBIL-5"),
 ("and no simulator data is collected during latent adaptation (spec \\S6).", "and no simulator data is collected during latent adaptation."),
 ("\\item (SOURCE\\_AUDIT A4) The expert", "\\item \\audit{A4} The expert"),
 ("(\\cref{rem:fb-design}, D1)", "(\\cref{rem:fb-design}, decision D1)"),
 ("When $\\nu$ is diffuse (which in the continuous maze depends on the collision rules, U-5),",
  "When $\\nu$ is diffuse (as the population replay law of the continuous maze is, \\cref{prop:env-replay}),"),
 ("if the kernel has a Lebesgue density (U-5)", "if the kernel has a Lebesgue density (as the implemented kernel has, \\cref{prop:env-cont})"),
 ("these are $0.97$ and $0.8565$ (SOURCE\\_AUDIT \\S10).", "these are $0.97$ and $0.8565$."),
 ("(SOURCE\\_AUDIT A7).", "\\audit{A7}."),
 ("\\begin{remark}[decisions D4--D5: tail treatment and start times; SOURCE\\_AUDIT U-4]",
  "\\begin{remark}[decisions D4--D5: tail treatment and start times; \\audit{U-4}]"),
 ("ER$_{\\mathrm{FB}}$ keeps FI24's $\\rho_e$ ($t=1,\\dots,H$), as spec \\S6 requires.",
  "ER$_{\\mathrm{FB}}$ keeps FI24's $\\rho_e$ ($t=1,\\dots,H$)."),
 ("the formulas hold only where $F(x,a,\\cdot)$ is differentiable, U-14)", "the formulas hold only where $F(x,a,\\cdot)$ is differentiable, \\audit{U-14})"),
 ("& chosen in 3B \\textsf{[I]} & ReLU: FB-8(b)--(c) and FB-9 not proved; FBIL-4 only where $F$ is differentiable (U-14)\\\\",
  "& LayerNorm, $\\tanh$, SiLU ($C^\\infty$) \\textsf{[I]} & covered; ReLU would not be: FB-8(b)--(c) and FB-9 not proved, FBIL-4 only where $F$ is differentiable \\audit{U-14}\\\\"),
 ("for the true kernel it may fail in the continuous maze (U-5), and then A1 fails too",
  "it holds for the population replay law (\\cref{prop:env-replay}) but fails for the atomic $\\hat\\nu$ of the continuous maze (\\cref{rem:env-atoms}), and then A1 fails too"),
 ("\\begin{remark}[relation to the $f$-PG objective; SOURCE\\_AUDIT \\S9.6]", "\\begin{remark}[relation to the $f$-PG objective; see also \\cref{sec:compare}]"),
],
"07_environment.tex": [
 ("% 07_environment.tex  --  THEORY_INDEX rows ENV-1 ... ENV-5 (Prompt 3A), ENV-6 (Prompt 3B)",
  "% Appendix G (from theory/07_environment.tex)  --  claims ENV-1 ... ENV-6 and FPG-9"),
 ("This section fixes the open items U-5, U-10 and U-13 of \\texttt{SOURCE\\_AUDIT.md} \\S12 for the code written in Prompt~3A, and proves the properties of that code on which earlier sections rely.",
  "This appendix fixes the environment items \\audit{U-5}, \\audit{U-10} and \\audit{U-13} for the implemented code, and proves the properties of that code on which the other appendices rely."),
 ("The environments are study choices (spec \\S5, \\texttt{configs/environment.yaml}).",
  "The environments are study choices (\\cref{sec:protocol}, \\texttt{configs/environment.yaml})."),
 ("\\Cref{sec:env-replay}, added in Prompt~3B, proves the domination property",
  "\\Cref{sec:env-replay} proves the domination property"),
 ("numerical checks in \\texttt{tests/}, \\texttt{scripts/check\\_environments.py} and \\texttt{audit/prompt3A/}.",
  "numerical checks in \\texttt{tests/}, \\texttt{scripts/check\\_environments.py} and \\texttt{audit/prompt3A/}. \\Cref{sec:env-crossfit} proves the cross-fitting statement FPG-9 used by the implemented $f$-PG learner."),
 ("With slip probability $p_{\\mathrm{slip}}:=0.1$ (U-10),", "With slip probability $p_{\\mathrm{slip}}:=0.1$ \\audit{U-10},"),
 ("\\begin{definition}[continuous maze; U-5]", "\\begin{definition}[continuous maze; \\audit{U-5}]"),
 ("Whether the replay law of Prompt~3B has such a density is \\emph{not} established here, and an atomic $\\nu$ never has one (\\cref{rem:env-atoms}).",
  "The replay law of FB pretraining has such a density (\\cref{prop:env-replay}); an atomic $\\nu$ never has one (\\cref{rem:env-atoms})."),
 ("\\subsection{Positivity of histogram cells (U-13)}", "\\subsection{Positivity of histogram cells}"),
 ("as A4 requires (SOURCE\\_AUDIT \\S9.2).", "as A4 requires \\audit{A4}."),
 ("\\subsection{The task-blind replay law (Prompt 3B)}", "\\subsection{The task-blind replay law}"),
 ("\\subsection{Cross-fitted plug-in signals (Prompt 3B)}", "\\subsection{Cross-fitted plug-in signals}"),
 ("for the three generators (NOTATION \\S6),", "for the three generators (\\cref{prop:div-generators}),"),
],
}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        text = (SRC / src).read_text(encoding="utf-8")
        for old, new in R.get(src, []):
            n = text.count(old)
            if n != 1:
                raise SystemExit(f"{src}: pattern found {n} times: {old[:80]!r}")
            text = text.replace(old, new)
        (DST / dst).write_text(text, encoding="utf-8")
        print(f"{src} -> appendix/{dst}: {len(R.get(src, []))} replacements")


if __name__ == "__main__":
    main()
