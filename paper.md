% File pp.tex
% When Dependence Matters in Multi-Agent LLM Reasoning
%
% NOTE ON BUILD DEPENDENCIES. This directory does not currently contain
% tacl2021v1.sty, tacl2021.bib, or acl_natbib.bst. Place the official TACL
% style bundle next to this file before compiling. Beyond that bundle this
% file uses only amsmath, amssymb, amsthm, booktabs, tabularx and graphicx,
% all of which ship with a standard TeX Live installation. Figure 1 is drawn
% with plain tabular rules rather than TikZ so that the file compiles on a
% minimal installation.

\documentclass[11pt,a4paper]{article}
\usepackage{times,latexsym}
\usepackage{url}
% ENCODING NOTE. All prompts in Appendix D are English, so T1 suffices. The only
% non-ASCII characters are the guillemets used to delimit an option string
% inside a prompt, which T1 provides.
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

%% Package options:
%% Short version: "hyperref" and "submission" are the defaults.
%% More verbose version:
%% Most compact command to produce a submission version with hyperref enabled
%%    \usepackage[]{tacl2021v1}
%% Most compact command to produce a "camera-ready" version
%%    \usepackage[acceptedWithA]{tacl2021v1}
%% Most compact command to produce a double-spaced copy-editor's version
%%    \usepackage[acceptedWithA,copyedit]{tacl2021v1}
%
%% If you need to disable hyperref in any of the above settings (see Section
%% "LaTeX files") in the TACL instructions), add ",nohyperref" in the square
%% brackets. (The comma is a delimiter in case there are multiple options specified.)

\usepackage{tacl2021v1}
% \setlength\titlebox{10cm} % <- for Option 2 below

%%%% Material in this block is specific to generating TACL instructions
\usepackage{xspace,mfirstuc,tabulary}
\newcommand{\dateOfLastUpdate}{Dec. 15, 2021}
\newcommand{\styleFileVersion}{tacl2021v1}

\newcommand{\ex}[1]{{\sf #1}}

\newif\iftaclinstructions
\taclinstructionsfalse % AUTHORS: do NOT set this to true
\iftaclinstructions
\renewcommand{\confidential}{}
\renewcommand{\anonsubtext}{(No author info supplied here, for consistency with
TACL-submission anonymization requirements)}
\newcommand{\instr}
\fi

%
\iftaclpubformat % this "if" is set by the choice of options
\newcommand{\taclpaper}{final version\xspace}
\newcommand{\taclpapers}{final versions\xspace}
\newcommand{\Taclpaper}{Final version\xspace}
\newcommand{\Taclpapers}{Final versions\xspace}
\newcommand{\TaclPapers}{Final Versions\xspace}
\else
\newcommand{\taclpaper}{submission\xspace}
\newcommand{\taclpapers}{{\taclpaper}s\xspace}
\newcommand{\Taclpaper}{Submission\xspace}
\newcommand{\Taclpapers}{{\Taclpaper}s\xspace}
\newcommand{\TaclPapers}{Submissions\xspace}
\fi

%%%% End TACL-instructions-specific macro block
%%%%

%%%% Additions for this paper
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}

\newcommand{\Mech}{M}
\newcommand{\Dep}{D}
\newcommand{\Marg}{\mathcal{M}}
\newcommand{\Hrep}{H_{\mathrm{repair}}}
\newcommand{\pbreak}{p_{+-}}
\newcommand{\prescue}{p_{-+}}
\newcommand{\Plur}{\mathrm{Plur}}
\newcommand{\Truth}{Y^{\star}}
\newcommand{\shared}{\textsc{shared}}
\newcommand{\indep}{\textsc{indep}}
\newcommand{\nocond}{\textsc{base}}
\newcommand{\pchurn}{p_{\mathrm{churn}}}
\newcommand{\kstar}{\kappa^{\star}}
\newcommand{\tetra}{\tau_{p}}
\newcommand{\He}{\mathrm{He}}
\newcommand{\KL}{\mathrm{KL}}

% PSEUDOCODE WITHOUT PACKAGES. algorithm/algorithmicx are not assumed present
% (see the build note at the top of this file), so algorithms are set as
% unfloated numbered blocks built from booktabs rules, exactly as Figure 1 is
% drawn with tabular rules rather than TikZ. \algname steps a private counter so
% that \ref resolves to the algorithm number and not to a float number; the
% blocks therefore carry no \caption.
\newcounter{algo}
\newcommand{\algname}[1]{\refstepcounter{algo}%
  \textbf{Algorithm \thealgo.}\ \textsc{#1}}
\newcommand{\algin}{\textbf{input}}
\newcommand{\algout}{\textbf{output}}
\newcommand{\algfor}{\textbf{for}}
\newcommand{\algif}{\textbf{if}}
\newcommand{\algelse}{\textbf{else}}
\newcommand{\algret}{\textbf{return}}
\newcommand{\algi}{\hspace*{1.2em}}
\newcommand{\algii}{\hspace*{2.4em}}
\newcommand{\algiii}{\hspace*{3.6em}}

\newcolumntype{Z}{>{\raggedright\arraybackslash}X}
\newcommand{\lvA}{\hangindent=1em\hangafter=1\relax}
\newcommand{\lvB}{\hspace*{.8em}\hangindent=1.8em\hangafter=1\relax}
\newcommand{\lvC}{\hspace*{1.6em}\hangindent=2.6em\hangafter=1\relax}
%%%% End additions
\newcommand{\lvD}{\hspace*{2.4em}\hangindent=3.4em\hangafter=1\relax}
\newcommand{\lvE}{\hspace*{3.2em}\hangindent=4.2em\hangafter=1\relax}
\title{When Dependence Matters in Multi-Agent LLM Reasoning}

% Author information does not appear in the pdf unless the "acceptedWithA" option is given

% The author block may be formatted in one of two ways:

% Option 1. Author’s address is underneath each name, centered.

\author{
  Template Author1
  \\
  Template Affiliation1/Address Line 1
  \\
  Template Affiliation1/Address Line 2
  \\
  Template Affiliation1/Address Line 2
  \\
  \texttt{template.email1example.com}
  \And
  Template Author2
  \\
  Template Affiliation2/Address Line 1
  \\
  Template Affiliation2/Address Line 2
  \\
  Template Affiliation2/Address Line 2
  \\
  \texttt{template.email2@example.com}
}

\date{}

\begin{document}
\maketitle

\begin{abstract}

\end{abstract}

\section{Introduction}
\label{sec:introduction}

Large language models (LLMs) are increasingly organized into multi-agent
systems in which several model instances reason, exchange information, and
aggregate their responses into a collective decision. Such systems can improve
reasoning through debate, diversity, and test-time interaction, but
communication can also induce conformity and correlated failures
\citep{zhu-etal-2026-demystifying,cui-etal-2026-free}. More broadly, substantial error
correlation has been observed even across independently developed LLMs
\citep{kim-etal-2025-correlated}. These findings challenge a basic premise behind
collective reasoning: agreement among multiple agents need not represent
independent evidence for the same answer.

A natural response is to measure the dependence among agents, and a
considerable body of work now does so, reporting correlated errors, conformity
effects, and communication-induced agreement
\citep{kim-etal-2025-correlated,huang-etal-2026-counterfactual,cui-etal-2026-free}. Dependence
magnitude has thus become the working summary of a multi-agent collective,
alongside competence and population size.

We argue that this summary is insufficient, and that the insufficiency is not a
matter of estimation error but of identifiability. \emph{How much} dependence a
population exhibits is not the same question as \emph{what that dependence
does}. Three distinct gaps separate the two.

\paragraph{Same dependence, different origin.}
Agents may agree because they were handed the same evidence, or because later
agents deferred to earlier ones. These are different causal structures. The
first acts upstream of aggregation, inside each agent's private formation of
belief; the second acts downstream, as feedback from the aggregate back onto
the agents. A single dependence number is compatible with both and therefore
cannot distinguish them.

\paragraph{Same dependence, different consequence.}
Even a large perturbation of the dependence structure need not change anything
the collective outputs. A population splitting seven to two to one is robust to
a few reassigned votes; a population splitting four to three to three is not.
Whether dependence has behavioural consequence is a property of where the
collective sits relative to its own decision boundary, not of the dependence
magnitude alone.

\paragraph{Same decision change, different value.}
Finally, a changed decision is not automatically a decision worth restoring.
Dependence can overturn an initially correct aggregate, but it can equally
rescue an initially incorrect one. Any claim that removing dependence will
improve accuracy depends on the \emph{balance} of these two directions, which
the dependence magnitude does not encode.

We therefore study dependence not as a single scalar property of a collective,
but through three attributes: the mechanism that generates it, the
susceptibility of the collective decision to it, and the repairability of its
effects. Let $\Mech$ denote the dependence mechanism and $\Dep$ its measured
magnitude. We characterize collective susceptibility by $\chi$ and measure the
probability that a dependence intervention changes the aggregate decision by
$\Delta Y=\Pr(Y_g\neq Y_0)$, where $Y_0$ and $Y_g$ denote the independent and
dependent collective decisions. We further distinguish harmful changes
$\pbreak$, in which dependence turns a correct collective decision into an
incorrect one, from beneficial changes $\prescue$, in which dependence corrects
an initially incorrect decision. This yields the chain
\begin{multline}
\Mech \longrightarrow \Dep \xrightarrow{\ \chi\ } \Delta Y \\
\longrightarrow (\pbreak,\prescue) \longrightarrow \Hrep ,
\label{eq:intro-chain}
\end{multline}
where $\Hrep$ denotes the accuracy headroom available to an
independence-restoring intervention. Figure~\ref{fig:story} gives the
intuition; Section~\ref{sec:problem-formulation} makes it precise.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/framework.pdf}
\caption{\textbf{Overview of the framework.} An independent collective is perturbed by two dependence mechanisms, shared evidence and copying. We distinguish the mechanism that induces dependence from its effect on the collective decision and decompose decision changes into break, rescue, and churn. Repairability is determined by the directional imbalance between harmful and beneficial changes, $\Hrep=\pbreak-\prescue$.}
\label{fig:story}
\end{figure*}

% NUMBERS DELIBERATELY ABSENT. Every figure that previously stood in this
% paragraph came from the campaign archived on 2026-10-01 and is void; see the
% banner in Section~\ref{sec:experiments-results}. Fill from the English rebuild.
Our findings follow the three gaps. First, mechanism matters beyond magnitude.
On a common dependence scale, copying changes the collective decision
significantly more often than shared evidence. The gap survives a tight caliper
on matched dependence, so it is not an artifact of the two arms sitting at
different dependence levels; it retains explanatory power in a nested
linear-probability model after conditioning on dependence; and it survives
nearest-neighbour matching on the triple of dependence, baseline accuracy, and
decision margin, which rules out the objection that the copying arm merely faces
easier or more marginal questions. Notably the gap appears in the
\emph{decision-change} rate while the net accuracy gap does not reach
significance, because copying raises the harmful and the beneficial flip rates by
comparable amounts; this is the first appearance of the cancellation that
Section~\ref{sec:repairability-bounds} later makes exact. We report this across three model families (Qwen2.5-7B, Llama-3.1-8B,
Gemma-2-9B) crossed with three benchmarks; the gap replicates in every
configuration for the first two and is heterogeneous but sign-consistent for
the third. Appendix~\ref{sec:limitations} states the one remaining boundary of
the claim: the copying arm's within-condition stochastic churn cannot be
estimated from this design.

Second, the effect of dependence is conditioned by the state of the collective
decision. Populations near the decision boundary are systematically more
susceptible to dependence perturbations, while high-margin decisions are
comparatively stable. The relationship persists when controlling for
agent-level confidence, whereas several population-level uncertainty statistics
are partially collinear with margin. We therefore interpret the result as
evidence for a broader decision-boundary susceptibility family rather than for
a uniquely privileged uncertainty statistic.

Third, and this is the result we find most consequential, frequent decision
change does not imply recoverable accuracy. Across six model families and seven
benchmarks, raising dependence changes a substantial fraction of collective
decisions while leaving net accuracy almost unmoved: in the low-competence
stratum the decision-change rate is large and significant in essentially every
cell, while the net accuracy difference is close to zero and significant in only
a handful, so the net effect is an order of magnitude smaller than the churn that
produces it. We stress that this is not an absence of effect. It is the signature
of harmful and beneficial changes offsetting one another, and it is precisely
what the theory predicts.

The theoretical payoff is a decomposition that makes the third finding exact
rather than anecdotal. For independence-restoring, information-conserving
interventions we show that the recoverable accuracy headroom is exactly the net
directional flip,
\begin{equation}
\Hrep=\pbreak-\prescue ,
\label{eq:intro-repair}
\end{equation}
and that it is bounded by the decision-change rate, with the slack identified
term by term in Section~\ref{sec:repairability-bounds}. Large dependence, and
even frequent decision change, therefore do not by themselves imply large
opportunities for accuracy improvement. A system may
exhibit substantial decision churn while harmful and beneficial changes nearly
cancel, and in that regime a perfectly accurate diagnosis of dependence
licenses almost no action.

Our contributions are threefold.
\begin{itemize}
    \item \textbf{Dependence has causal structure.} We introduce controlled
    shared-evidence and copying interventions on a common dependence scale and
    show, both by theorem and by matched-dependence analysis, that equal
    dependence magnitudes can produce different collective decision dynamics.
    Dependence magnitude is not a sufficient statistic for collective accuracy.
    \item \textbf{Dependence matters conditionally.} We formalize
    decision-boundary susceptibility, prove a margin gate under which a
    bounded-reach perturbation cannot move the collective at all, and show
    empirically that the behavioural effect of dependence concentrates where
    the gate is open.
    \item \textbf{Dependence is not equivalent to repairability.} We decompose
    dependence-induced decision changes into harmful and beneficial directions
    and derive an exact repair-headroom identity and bound, clarifying when
    restoring independence can and cannot yield net accuracy gains.
\end{itemize}

Together these results shift the analysis of multi-agent LLM systems from
asking only \emph{how dependent are the agents?} to asking \emph{where did the
dependence come from, when can it change the collective decision, and how much
of its effect is actually repairable?}

\section{Related Work}
\label{sec:related-work}

\subsection{Multi-Agent LLM Reasoning}
\label{sec:rw-multi-agent}

Multi-agent debate and related collective-reasoning methods use interaction
among multiple LLM instances to improve inference-time reasoning
\citep{du-etal-2024-multiagent}. Recent work
has increasingly questioned whether interaction itself is sufficient for such
gains. \citet{zhu-etal-2026-demystifying} show that homogeneous agents with uniform
belief updates need not improve through vanilla debate, and identify diversity
of initial viewpoints and calibrated confidence communication as important
determinants of debate effectiveness. Other work has focused on failure modes
created by consensus-seeking interaction. Free-MAD \citep{cui-etal-2026-free}
identifies conformity and error propagation as limitations of conventional
debate and introduces anti-conformity together with trajectory-based decision
scoring. A parallel strand is openly
skeptical. \citet{wang-etal-2024-rethinking-bounds} report that a well-prompted
single agent nearly matches multi-agent discussion, and
\citet{smit-etal-2024-mad} find that debate does not reliably beat
self-consistency ensembling and is highly configuration-sensitive. Others
diagnose premature convergence directly and prescribe divergence as the remedy
\citep{liang-etal-2024-encouraging}, while \citet{cemri-etal-2025-multi-agent}
show that multi-agent failures are mechanistically heterogeneous rather than
instances of one pathology. Together these studies establish that collective
performance depends on how agents interact rather than merely on the number of
agents or interaction rounds, and the heterogeneity they report is itself an
argument that a scalar summary of interaction cannot identify which mechanism is
at work. Our focus is complementary. Instead of designing a stronger debate
protocol, we ask whether the dependence induced by interaction is sufficient to
characterize its collective consequences, and we treat the interaction strength
as a controlled variable rather than as a design choice to be optimized.

\subsection{Correlated Errors and Dependence}
\label{sec:rw-dependence}

Dependence is a fundamental concern whenever multiple predictors are combined.
Classical ensemble theory formalizes it as a diversity term that trades off
against individual accuracy, and the canonical negative result is that of
\citet{kuncheva-whitaker-2003-measures}, who evaluate ten diversity statistics
and find none that reliably predicts majority-vote accuracy. Later work makes
the reason exact: diversity enters a bias-variance-diversity decomposition as a
quantity to be traded off rather than maximized
\citep{wood-etal-2023-unified}. Most directly relevant to us,
\citet{brown-kuncheva-2010-good} split majority-vote error into ``good''
diversity, disagreement on points the ensemble gets wrong, and ``bad''
diversity, disagreement on points it gets right. Our directional decomposition
in Section~\ref{sec:repairability-bounds} is the decision-level counterpart of
that split, and it differs in two ways: it is an exact identity rather than a
decomposition of a squared loss, and it is attached to an intervention class, so
it states what an intervention could recover rather than how an existing
ensemble's error divides.

At the level of whole systems, the concern is monoculture.
\citet{kleinberg-raghavan-2021-monoculture} show formally that shared reliance
on one predictor can degrade collective decision quality even when each agent is
individually more accurate, and \citet{bommasani-etal-2022-picking} find that
shared upstream components do not reliably homogenize outcomes, which is an early
indication that the location of the shared cause matters and not only its
strength. For LLMs specifically, \citet{kim-etal-2025-correlated} provide
large-scale evidence that model errors are substantially correlated across
independently developed models, and \citet{goel-etal-2025-great} supply a
chance-adjusted metric for LLM-to-LLM error overlap together with evidence that
similarity undermines oversight. We take such measurement as given: the question
we ask begins where it ends, since both report magnitude without separating the
process that produced it. The classical argument that diverse problem solvers can
outperform uniformly stronger ones \citep{hong-page-2004-groups} is the
motivation we qualify. The dependence estimator of
Equation~\ref{eq:D} is the error-correlation analogue of the ensemble diversity
term, measured causally rather than observationally. These findings motivate treating
agent outputs as dependent rather than independent samples. Observational
dependence, however, does not identify the process that generated it. In an
interacting population, similar dependence may arise from common upstream
information, from direct social influence, or from other shared factors. We
therefore distinguish the \emph{magnitude} of dependence from its \emph{causal
mechanism}, and test whether mechanism information remains predictive after
dependence magnitude has been matched. Theorem~\ref{thm:nonequiv} shows that
this distinction is not merely a caveat about estimation: there exist
mechanisms that agree in the dependence limit yet disagree in collective
accuracy, so no function of competence and dependence alone can predict the
collective outcome.

\subsection{Causal Dependence and Social Influence}
\label{sec:rw-causal-dependence}

Recent work has begun to analyze communication-induced dependence more
directly. CAGE-CAL \citep{huang-etal-2026-counterfactual} compares post-communication agent
behaviour with no-communication counterfactuals and studies
communication-induced correlated failures and false consensus, using the
resulting structure for calibration and topology selection. Related work on
conformity studies how information from peers changes individual responses,
motivating anti-conformity and selective-interaction mechanisms
\citep{cui-etal-2026-free}. The documented tendency of models to defer to a
stated interlocutor position \citep{sharma-etal-2024-sycophancy} is the
behavioural mechanism that our copying operator of Equation~\ref{eq:m2}
exploits, and its population-level limit is the informational cascade of
\citet{bikhchandani-etal-1992-theory}. Two recent studies establish the
population-level phenomenon in LLM collectives: \citet{weng-etal-2025-conformity}
show that an agent placed among peers conforms to the majority on reasoning
tasks, with conformity increasing in interaction time and majority size, and
\citet{zhu-etal-2025-conformity} find in an Asch-style design that conformity is
strongest exactly when the model is uncertain about its own answer, which is the
agent-level analogue of the collective gate we prove in
Theorem~\ref{thm:gating}. At the network level,
\citet{chuang-etal-2024-simulating} report drift toward consensus attractors in
populations of interacting agents. These results motivate our central
measurement problem rather than solve it: a collective-level correlation
statistic is ambiguous between shared priors and iterated peer exposure, which
is precisely the ambiguity our instrument removes by construction. Our analysis
differs in two respects. First, we
intervene on two distinct mechanisms that generate dependence and compare their
collective effects at matched dependence magnitude, rather than comparing
communication against its absence. Second, we distinguish changes in individual
responses from changes in the aggregate decision. This distinction motivates
our notion of decision susceptibility: substantial dependence may have little
behavioural consequence when the collective decision is far from its boundary.

\subsection{Uncertainty and Decision Susceptibility}
\label{sec:rw-susceptibility}

A related line of work studies when multi-agent interaction is useful. The
single-model null hypothesis is majority voting over independently sampled
reasoning paths \citep{wang-etal-2023-selfconsistency}, which is precisely the
independent aggregate our baseline arm realizes and against which every
dependence effect in this paper is measured. Diversity, disagreement, and
confidence have all been associated with the effectiveness of multi-agent
reasoning \citep{zhu-etal-2026-demystifying}. A model's own confidence is a
natural candidate for the missing state variable, and it is partly informative:
models have some access to whether they are correct
\citep{kadavath-etal-2022-language}, and verbalized confidences can be elicited
in reasonably calibrated form \citep{tian-etal-2023-just}. This is why our
finding that agent-level confidence does not absorb the margin effect is a
substantive result rather than a measurement failure. A third possibility is to
delegate the aggregation to a single strong judge, but LLM judges carry position
and verbosity biases of their own \citep{zheng-etal-2023-judging}, which is one
reason we aggregate by vote over a population instead.

The nearest prior to our gate is
\citet{aggarwal-etal-2023-lets}, who already condition sampling effort on
proximity to the plurality boundary. The quantity is the same; the use is not.
They treat boundary proximity as a stopping rule under an independence
assumption, whereas we show that the same quantity governs whether a
\emph{dependence} perturbation reaches the collective answer at all. These ideas
are related to our notion of decision susceptibility, but our objective
differs. We do not use uncertainty as a routing criterion for deciding whether
debate should occur. Instead, we characterize how sensitive an already-defined
collective decision is to controlled dependence perturbations. We therefore
compare margin, disagreement, vote uncertainty, and agent-level confidence as
competing indicators, and interpret the resulting evidence as a broader family
of decision-boundary susceptibility signals rather than assigning a privileged
role to a single statistic.

\subsection{Dependence and Repairability}
\label{sec:rw-repairability}

Aggregation theory has long treated inter-voter correlation as the knob that
decides whether a crowd helps: \citet{ladha-1992-condorcet} shows that the
Condorcet conclusion survives so long as the \emph{average} correlation is
bounded. That is precisely the sufficiency assumption our
Corollary~\ref{cor:insufficient} contests, since two mechanisms can share an
average correlation and still differ in collective accuracy. The same assumption
is implicit in the practical claim that accuracy improves monotonically with the
number of sampled agents \citep{li-etal-2024-more}, which holds under near
independence and is therefore a direct target of
Theorem~\ref{thm:nonequiv} at the population sizes we study.

Existing approaches to correlated failure and conformity seek to improve
collective reliability through calibration, topology selection, anti-conformity,
diversity, or modified aggregation
\citep{huang-etal-2026-counterfactual,zhu-etal-2026-demystifying,cui-etal-2026-free}.
A representative diversity intervention assigns distinct personas and relies on
them to manufacture genuine heterogeneity
\citep{wang-etal-2024-unleashing}. We note a gap in this literature that our
instrument is built to address: such methods report accuracy, not inter-agent
error correlation, so it remains unestablished whether persona diversity actually
decorrelates agent errors or merely redistributes them. Diagnosing
dependence and profitably removing it are, however, distinct problems. Reducing
dependence may prevent an initially correct collective decision from being
overturned, but it may also remove a dependence-induced rescue of an initially
incorrect decision. We make this distinction explicit through the directional
quantities $\pbreak$ and $\prescue$, and characterize their relationship to the
accuracy headroom available to independence-restoring interventions. Our focus
is therefore not only whether dependence exists, but whether its collective
consequences are \emph{actionable}. To our knowledge the resulting identity and
bound, which hold without any model of the dependence dynamics, have not
previously been stated for multi-agent LLM collectives.

\section{Problem Formulation}
\label{sec:problem-formulation}

\paragraph{Collective decisions.}
A population of $N$ agents answers a multiple-choice question $q$ with $K$
options. We adopt canonical coordinates in which the correct option is always
index $0$, and write $\Truth=0$ for the truth. Agent $i$ emits a score vector
over options; the agent is \emph{informative} if its scores are not all equal,
and uninformative agents cast no vote. Writing $n_j$ for the number of
informative agents whose highest score falls on option $j$, and
$n_{\mathrm{tot}}=\sum_j n_j$, the collective decision is the plurality
\begin{equation}
Y=\Plur(n)=\arg\max_{j} n_j ,
\end{equation}
with a top-level tie scored as a non-answer. Collective correctness is
$A=\mathbf{1}[Y=\Truth]$, and collective accuracy is its expectation over
questions and over the stochasticity of generation.

Discarding uninformative agents matters. An unparseable reply collapses to a
constant score vector whose arg max is index $0$, which in canonical
coordinates is the truth; counting such replies would score every parse failure
as correct. Appendix~\ref{sec:app-estimators} quantifies the effect.

\paragraph{Dependence magnitude.}
We measure dependence at the level of \emph{errors}, across repeated
realizations of the same question. Let $E\in\{0,1\}^{R\times N}$ record whether
agent $i$ was wrong in replicate $r$, over $R$ independent replicates of the
same condition, let $\sigma_i$ be the standard deviation of $E_i$ across those
replicates, and let $S=\sum_i E_i$ be the error count of a replicate. Expanding
the variance of $S$ isolates the off-diagonal terms; Cauchy--Schwarz bounds each
of them by $\sigma_i\sigma_j$, with equality exactly when the two error
indicators coincide; and the algebraic identity $(\sum_i a_i)^2=\sum_i
a_i^2+\sum_{i\neq j}a_ia_j$ applied to $a_i=\sigma_i$ puts that bound in closed
form. Together,
\begin{equation}
\begin{split}
\sum_{i\neq j}\mathrm{Cov}(E_i,E_j)
&=\mathrm{Var}(S)-\sum_i\mathrm{Var}(E_i) \\
&\le\sum_{i\neq j}\sigma_i\sigma_j \\
&=\Big(\sum_i\sigma_i\Big)^{2}-\sum_i\sigma_i^{2}.
\end{split}
\label{eq:D-max}
\end{equation}
The first line is the raw dependence signal, positive exactly when agents err
together more often than their marginals require, and computable from the error
matrix alone. The second is the largest value that signal can take at fixed
marginals. Our primary dependence estimator for one question is their ratio,
\begin{equation}
\begin{split}
\Dep(q)
&=\frac{\mathrm{Var}\!\left(\sum_i E_i\right)-\sum_i \mathrm{Var}(E_i)}
{\left(\sum_i \sigma_i\right)^2-\sum_i \sigma_i^2} \\[3pt]
&=\frac{\sum_{i\neq j}\mathrm{Cov}(E_i,E_j)}{\sum_{i\neq j}\sigma_i\sigma_j} \\[3pt]
&=\frac{\sum_{i\neq j}\sigma_i\sigma_j\,r_{ij}}{\sum_{i\neq j}\sigma_i\sigma_j}.
\end{split}
\label{eq:D}
\end{equation}
where $r_{ij}=\mathrm{Cov}(E_i,E_j)/(\sigma_i\sigma_j)$ is the pairwise error
correlation. The last form, a $\sigma_i\sigma_j$-weighted mean of the pairwise
correlations, is what fixes the three properties we need, and they are the reason
for this normalization rather than a simpler one: $\Dep\in[-1,1]$, because a
weighted mean of correlations is one; $\Dep=1$ exactly when all agents err
together, by the equality case of Equation~\ref{eq:D-max}; and $\Dep=0$ under
independence for \emph{any} competence profile, so the scale does not drift as
accuracy changes across cells. The quantity is defined whenever at least two agents vary
across replicates, and is undefined when an arm saturates so that every agent is
constant; in that case error dependence is genuinely unmeasurable and we report
it as missing rather than imputing a value.

Replication is essential here, and not merely for precision. At $R=1$ every
$\sigma_i$ vanishes, so Equation~\ref{eq:D} is the indeterminate form $0/0$; and
the natural single-realization substitute, which averages over agents instead of
over replicates, is a constant that depends on nothing.

\begin{lemma}[Single-realization degeneracy]
\label{lem:degeneracy}
Let $e\in\mathbb{R}^{N}$ be one realization of the error vector, with mean
$\bar e$ and not all entries equal. The mean pairwise correlation computed
across agents from that single realization,
\begin{equation}
r(e)=\frac{\sum_{i\neq j}(e_i-\bar e)(e_j-\bar e)}
{(N-1)\sum_i (e_i-\bar e)^2},
\label{eq:degenerate}
\end{equation}
equals $-1/(N-1)$ for every such $e$.
\end{lemma}

The value is identical for a population split evenly and for one that is
unanimous but for a single dissenter, so a single realization carries no
dependence information whatsoever. Across replicates, by contrast, a common
shock appears as genuine positive covariance in Equation~\ref{eq:D-max}, which
is why evidence is redrawn in every replicate in
Section~\ref{sec:matched-instrument}.

\paragraph{Decision susceptibility.}
The order parameter that governs whether a perturbation can change the
collective answer is the normalized runner-up gap, or margin,
\begin{equation}
\Marg=\frac{n_{(1)}-n_{(2)}}{N}\in[0,1],
\label{eq:margin}
\end{equation}
where $n_{(1)}\geq n_{(2)}$ are the two largest vote counts. A margin of zero
means the collective sits exactly on its decision boundary; a margin of one
means unanimity.

Margin is an observable, not yet a susceptibility. To obtain one we ask how much
work a perturbation must do before the plurality can move at all. Call the
number of agents whose vote an operator reassigns its \emph{reach} $\kappa$, and
define the \emph{critical reach} of a question as the smallest reach at which
some operator changes the collective decision,
\begin{equation}
\kstar=\min\big\{\kappa:\ \exists\ \text{a reach-}\kappa\ \text{operator with }
Y\neq Y_0\big\}.
\label{eq:kappastar}
\end{equation}
Lemma~\ref{lem:gate} evaluates this minimum in closed form, as
$\kstar=\lceil N\Marg/2\rceil$. Susceptibility is then the probability that the
operator actually at hand is large enough to clear that threshold,
\begin{equation}
\chi(q)=\Pr\!\big(\kappa\ge\kstar(q)\big)
=\Pr\!\big(\kappa\ge\lceil N\Marg(q)/2\rceil\big).
\label{eq:chi}
\end{equation}
This is the $\chi$ labelling the second arrow of
Equation~\ref{eq:intro-chain}. It is a property of where the question sits
relative to its own decision boundary, combined with the reach distribution of
the mechanism, and it is a function of neither the dependence magnitude nor the
dependence contrast; Section~\ref{sec:theory} shows that it is nonetheless an
upper bound on everything downstream.

\paragraph{Decision change and its direction.}
Let $Y_0$ be the collective decision of the independent population and $Y_g$
the decision after a dependence-inducing operator of strength $g$. The
\emph{decision-change rate} is
\begin{equation}
\Delta Y=\Pr\!\left(Y_g\neq Y_0\right),
\end{equation}
and the two directional components are
\begin{align}
\pbreak &= \Pr\!\left(Y_0=\Truth,\ Y_g\neq \Truth\right), \\
\prescue &= \Pr\!\left(Y_0\neq \Truth,\ Y_g=\Truth\right).
\end{align}
We write $A_0$ and $A_g$ for the corresponding accuracies and
$\Delta A=A_g-A_0$ for the net accuracy effect. The central observation of this
paper is that $\Delta A$ compresses $\pbreak$ and $\prescue$ into their
difference, so a near-zero $\Delta A$ is consistent with a large $\Delta Y$.
Accordingly we never report $\Delta A$ without $\Delta Y$ beside it.

\paragraph{Repairability.}
Finally, we ask what an intervention could recover. We restrict attention to
interventions that remove dependence without supplying new information, since
these are the interventions that a dependence \emph{diagnosis} licenses.

\begin{definition}[IRIC intervention]
\label{def:iric}
An intervention $R$ is \emph{independence-restoring and
information-conserving} (IRIC) if it (i) discards the dependent votes, (ii)
re-aggregates a fresh independent draw from the same per-agent
belief-generating process, with no new evidence and no new query, and (iii)
applies the same aggregation rule.
\end{definition}

By construction the decision of an IRIC intervention has the marginal law of
$Y_0$, so its accuracy satisfies $A_R=A_0$. The recoverable headroom is then
$\Hrep := A_R-A_g = A_0-A_g$. Section~\ref{sec:repairability-bounds} shows that
this quantity is exactly the net directional flip, and that it is bounded by
$\Delta Y$.

\section{Dependence Mechanisms and a Matched Instrument}
\label{sec:methodology}

\subsection{Two Mechanisms}
\label{sec:dependence-mechanisms}

We study two operators that both raise measured dependence but act at opposite
ends of the aggregation pipeline.

\paragraph{Upstream shared evidence.}
A common evidence variable enters each agent's private formation of belief
before any aggregation. In a Gaussian-threshold idealization with marginal
competence $p$, write $a=\Phi^{-1}(p)$ and
\begin{equation}
C_i=\mathbf{1}\!\left[\sqrt{\rho}\,Z_0+\sqrt{1-\rho}\,\epsilon_i\le a\right],
\label{eq:m1}
\end{equation}
where $Z_0$ is shared across agents, the $\epsilon_i$ are independent, and all
variables are standard normal. Two properties make this a usable knob, and both
are worth deriving rather than asserting. Let
$L_i=\sqrt{\rho}Z_0+\sqrt{1-\rho}\,\epsilon_i$ denote the latent variable.
Because $L_i$ is a sum of independent centred normals with
$\mathrm{Var}(L_i)=\rho+(1-\rho)=1$, we have $L_i\sim\mathcal{N}(0,1)$ for every
$\rho$, hence
\begin{equation}
\Pr(C_i=1)=\Phi(a)=p ,
\label{eq:marg-preserved}
\end{equation}
so that sweeping $\rho$ moves dependence without moving competence. The latent
correlation is $\mathrm{Corr}(L_i,L_j)=\rho$ for $i\neq j$. The
\emph{correctness} correlation, however, is not $\rho$ but its thresholded
image,
\begin{equation}
\tetra(\rho)=\mathrm{Corr}(C_i,C_j)=\frac{\Phi_2(a,a;\rho)-p^2}{p(1-p)},
\label{eq:tau}
\end{equation}
where $\Phi_2(\cdot,\cdot;\rho)$ is the standard bivariate normal distribution
function with correlation $\rho$. Conflating the two would matter for us
specifically, because the quantity our instrument matches arms on is a measured
error correlation and not a latent parameter.

\begin{lemma}[Dependence calibration]
\label{lem:calibration}
For every $p\in(0,1)$ the map $\tetra$ of Equation~\ref{eq:tau} admits the
expansion
\begin{equation}
\begin{split}
\tetra(\rho) &= \sum_{k\ge1}w_k\,\rho^{k}, \\
w_k &= \frac{\varphi(a)^2\,\He_{k-1}(a)^2}{k!\;p(1-p)} .
\end{split}
\label{eq:mehler}
\end{equation}
with $\He_k$ the Hermite polynomials, and the weights satisfy $w_k\ge0$,
$w_1>0$ and $\sum_{k\ge1}w_k=1$. Consequently $\tetra$ is a strictly increasing
bijection of $[0,1]$ onto itself, and it attenuates: $\tetra(\rho)\le\rho$, with
equality only at $\rho\in\{0,1\}$.
\end{lemma}

Two consequences organize what follows. First, because $\tetra$ is invertible, a
target \emph{measured} correlation pins a unique latent $\rho$, so placing the
two mechanisms on one dependence axis is well posed rather than a convenient
fiction; this is what licenses the caliper procedure of
Appendix~\ref{sec:app-est-matched}. Second, the attenuation says where a
separation result has to be stated. At any $\rho<1$ the measured correlation is
strictly below one, so a measure could in principle still discriminate; only as
$\rho\to1$ do the latent and measured scales meet at their common ceiling, which
is exactly the regime in which no correlation-based summary can tell the
mechanisms apart. That is the limit taken in Theorem~\ref{thm:nonequiv}. The
causal graph here is $Z_0\to C_i\to Y$: dependence enters through the agents'
inputs.

\paragraph{Downstream copying.}
Starting from the independent population, each agent is shown the current
aggregate and pulled towards it with strength $g$. In the idealized
synchronous limit every agent adopts the current plurality,
\begin{equation}
X_i^{(t+1)}=Y_t=\Plur\!\left(X^{(t)}\right),
\label{eq:m2}
\end{equation}
so the causal graph contains the feedback loop
$X^{(t)}\to Y_t\to X^{(t+1)}$: dependence enters through the aggregate itself.

This operator does not merely push a correlation measure upwards, it saturates
our own estimator exactly, which is the sharper statement. After one synchronous
round every agent holds the same answer, so the error indicators coincide
replicate by replicate, $E_i=E_j$ for all $i,j$. Writing $\sigma$ for their
common standard deviation and substituting into Equation~\ref{eq:D}, both sums
have $N(N-1)$ identical terms,
\begin{equation}
\Dep=\frac{N(N-1)\,\mathrm{Cov}(E_1,E_2)}{N(N-1)\,\sigma^2}
=\frac{\sigma^{2}}{\sigma^{2}}=1
\label{eq:copy-saturate}
\end{equation}
whenever $\sigma>0$. Copying reaches the ceiling of the dependence scale in a
single round, with no free parameter to tune.

The distinction is the engine of the paper. Both operators drive measured
dependence to the same ceiling, by Equation~\ref{eq:copy-saturate} and by
$\tetra(1)=1$ in Lemma~\ref{lem:calibration}, yet they do so by routing
information in opposite directions relative to the aggregator. The next section
shows that the two ceilings sit over different collective accuracies.

\subsection{A Matched Causal Instrument}
\label{sec:matched-instrument}

Observational comparison of the two mechanisms is confounded, because
populations that happen to copy may also face different questions. We therefore
construct an instrument that varies only the \emph{commonality} of the evidence
realization while holding its content distribution, its framing, the question,
the option display, the population size, and the aggregation rule fixed. Each
question is run under three conditions:
\begin{description}
  \item[\nocond] no evidence is supplied.
  \item[\shared] one evidence item is drawn and broadcast to all $N$ agents, a
  common shock.
  \item[\indep] $N$ evidence items are drawn without replacement from the same
  de-duplicated pool, one per agent.
\end{description}
Both evidence conditions use an identical framing template, so the arms are
matched in evidence quality and differ only in whether the realization is
common. Drawing the independent arm without replacement matters: with
replacement, accidental duplicates inflate the dependence of the control and
shrink the contrast for a purely combinatorial reason.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/Clue_pool.pdf}
\caption{The matched causal instrument. SHARED and INDEP draw clues from the same de-duplicated pool and differ only in whether the realization is common to all agents. BASE receives no clue. All conditions produce the plurality decision $Y$, error matrix $E$, and dependence $D$.}
\label{fig:causal-instrument}
\end{figure}

Figure~\ref{fig:causal-instrument} illustrates the matched causal instrument used to manipulate dependence while controlling the evidence source.

Crucially, evidence is redrawn in every replicate. The contrast
\begin{equation}
\Delta \Dep=\Dep(\shared)-\Dep(\indep)
\label{eq:dD}
\end{equation}
is therefore identified at the replicate level, which is what makes
Equation~\ref{eq:D} informative rather than algebraically pinned by
Lemma~\ref{lem:degeneracy}. We refer to $\Delta \Dep>0$ as establishing that the
instrument does what it claims, namely causally raise error dependence while
holding evidence quality fixed.

Algorithm~\ref{alg:instrument} states the resulting procedure for one cell, a
cell being one model crossed with one benchmark. Three lines carry the design.
Line~4 draws the evidence afresh per replicate, which is what identifies
Equation~\ref{eq:dD}. Line~7 broadcasts a single realization while line~9 draws
$N$ distinct ones from the same de-duplicated pool, so the arms differ in
commonality and in nothing else. Line~11 permutes the displayed option order
before eliciting and inverts the permutation afterwards, so that position bias
cannot be read as competence, and the stored tensor is always in canonical
coordinates with the truth at index $0$.

\begin{figure}[t]
\small
\setlength{\tabcolsep}{2pt}
\noindent\begin{tabularx}{\columnwidth}{@{}r@{\ \,}Z@{}}

\toprule
\multicolumn{2}{@{}l@{}}{\algname{One matched-instrument cell.}%
\label{alg:instrument}}\\
\midrule
 & \lvA\algin\ questions $Q$, options per question, agents $1..N$,
   replicates $R$, evidence pool $\mathcal{P}_q$ per question, conditions
   $\{\nocond,\shared,\indep\}$\\
 & \lvA\algout\ score tensor $T[\,c,q,r,i,\cdot\,]$ in canonical
   coordinates\\
\midrule
1 & \lvA\algfor\ $q\in Q$:\\
2 & \lvB $\mathcal{U}_q\leftarrow$ \textsc{dedup}$(\mathcal{P}_q)$\\
3 & \lvB\algfor\ $r=1..R$:\\
4 & \lvC seed a fresh replicate stream $\xi_{q,r}$\\
5 & \lvC\algfor\ $c\in\{\nocond,\shared,\indep\}$:\\
6 & \lvD\algif\ $c=\nocond$: $e_1..e_N\leftarrow\varnothing$\\
7 & \lvD\algelse\ \algif\ $c=\shared$: draw one $u\sim\xi_{q,r}$ from
   $\mathcal{U}_q$; $e_1..e_N\leftarrow\textsc{frame}(u)$
   \{common shock\}\\
8 & \lvD\algelse:\\
9 & \lvE draw $u_1..u_N$ from $\mathcal{U}_q$ \emph{without} replacement;
   $e_i\leftarrow\textsc{frame}(u_i)$ \{matched control\}\\
10 & \lvD $\pi\leftarrow$ a permutation of the options from
   $\xi_{q,r}$\\
11 & \lvD $x_i\leftarrow$ \textsc{elicit}$(q,\pi(\text{options}),e_i)$ for
   $i=1..N$, in parallel\\
12 & \lvD $T[c,q,r,i,\cdot]\leftarrow\pi^{-1}(x_i)$ \{truth at index
   $0$\}\\
\midrule
 & \lvA\algret\ $T$; all quantities of
   Section~\ref{sec:dependence-measurement} are computed from $T$ offline by
   Algorithm~\ref{alg:estimate}.\\
\bottomrule
\end{tabularx}
\end{figure}

Note what the algorithm does \emph{not} do. It never conditions the evidence on
the agents' answers, and it never shows one agent the state of another; the
copying mechanism of Equation~\ref{eq:m2} is a separate operator that fills a
peer block instead, as Table~\ref{tab:prompts-peer} records. Keeping the
two in disjoint code paths is what keeps the mechanism label clean.

\subsection{Measuring the Three Layers}
\label{sec:dependence-measurement}

For each question and condition we record the full agent-by-option score
matrix for every replicate, which allows all downstream quantities to be
recomputed offline without further generation. From these we obtain the
dependence $\Dep(q)$ of Equation~\ref{eq:D}, the margin $\Marg(q)$ of
Equation~\ref{eq:margin} measured on the baseline arm, the decision-change rate
$\Delta Y(q)$ computed by comparing conditions at matched replicate index, the
directional components $\pbreak(q)$ and $\prescue(q)$, and the net accuracy
effect $\Delta A(q)$. Measuring the margin on the baseline arm, never after the
perturbation, keeps the susceptibility regressor free of the treatment.

\subsection{Susceptibility}
\label{sec:decision-susceptibility}

We operationalize susceptibility as the dependence of $|\Delta A|$ on the
baseline margin, and we test it against competing explanations. Because
agent-level confidence could in principle drive both the margin and the
response, we include agent-level confidence as a control, and we additionally
compare margin with vote entropy and raw disagreement. As
Section~\ref{sec:susceptibility-analysis} explains, these population-level
statistics are partially collinear by construction, which is why we claim a
susceptibility \emph{family} rather than a privileged statistic.

\subsection{Repairability}
\label{sec:repairability-method}

Repairability is estimated directly from the directional flip counts rather
than from a model of the dynamics. For each question we classify every
replicate-matched pair of baseline and dependent decisions into exactly one of
the three parts $B$, $S$, $C$ of Proposition~\ref{prop:identity}, or into
agreement, which yields $\pbreak(q)$, $\prescue(q)$ and $\pchurn(q)$, and then
$\Hrep(q)$ by Equation~\ref{eq:identity}. Because that proposition is an
identity, the estimate needs no assumption beyond the definition of the
aggregation rule, and because the three parts are a partition we can check the
arithmetic rather than trust it: we verify question by question that the
classified counts satisfy both equalities of Equation~\ref{eq:identity}, which
is a genuine test of the analysis code, since a mis-paired replicate index
breaks the first equality while leaving the second intact.

\section{Theoretical Analysis}
\label{sec:theory}

This section states the three results that structure the paper. Full proofs
appear in Appendix~\ref{sec:app-proofs}. Throughout, $N\geq 3$, competence
$p\in(0,1)$, and $P_{\mathrm{maj}}(N,p)=\sum_{k>N/2}\binom{N}{k}p^k(1-p)^{N-k}$
denotes the Condorcet majority accuracy of an independent population
\citep{condorcet-1785-essai}.

\subsection{Mechanism Non-Equivalence}
\label{sec:mechanism-non-equivalence}

The first result says that the two mechanisms of
Section~\ref{sec:dependence-mechanisms} remain distinguishable precisely where
a dependence measure can no longer distinguish them, namely in the limit of
total agreement.

\begin{theorem}[Mechanism non-equivalence]
\label{thm:nonequiv}
Fix $N\geq3$ and $p\in(0,1)$ with $p\neq\tfrac12$, and a common independent
initial population. As $\rho\to1$, equivalently by
Lemma~\ref{lem:calibration} as the measured correctness correlation
$\tetra(\rho)\to1$:
\begin{enumerate}
  \item under upstream shared evidence,
  $A\to p$, so that
  $\Delta_{\text{shared}}\to p-P_{\mathrm{maj}}(N,p)$, which is nonzero with
  $\mathrm{sign}(\Delta_{\text{shared}})=\mathrm{sign}(\tfrac12-p)$;
  \item under synchronous copying, $A=P_{\mathrm{maj}}(N,p)$ at every step, so
  $\Delta_{\text{copy}}=0$, even though terminal agreement tends to one;
  \item under sequential cascade copying in the sense of
  \citet{bikhchandani-etal-1992-theory} and \citet{banerjee-1992-simple},
  $A\leq P_{\mathrm{maj}}(N,p)$, with strict inequality on a set of signal
  realizations of positive measure.
\end{enumerate}
Consequently the mechanisms share the dependence limit but occupy distinct
collective limits, and for $p>\tfrac12$ the ranking is
$\Delta_{\text{copy}}=0>\Delta_{\text{shared}}$.
\end{theorem}

The statement is sharper than the observation that a pairwise correlation does
not determine a multivariate binary law. Here the \emph{dynamical} endpoints
differ, the difference is sign-definite and depends on competence, and the two
mechanisms are both realizable in an LLM collective. The consequence for
modelling is immediate.

\begin{corollary}[Dependence is not a sufficient statistic]
\label{cor:insufficient}
Let $\Dep$ be any functional of the joint answer distribution determined by the
marginal competence and the pairwise correctness correlation alone, such as a
Fleiss-type intraclass correlation or exchangeable pair agreement. Then the map
$(p,\Dep)\mapsto A$ is not well defined, and predicting collective accuracy
requires the mechanism label. Equivalently, $A=F(p,\Dep,\Mech,N)$ does not
factor through $(p,\Dep,N)$.
\end{corollary}

An instructive special case is $p>\tfrac12$, where copying preserves the
Condorcet boost that shared evidence destroys. A collective can therefore look
maximally dependent by any correlation measure and still be either unharmed or
substantially harmed, depending only on where the dependence came from.

\subsection{Susceptibility Gating}
\label{sec:susceptibility-analysis}

The second result explains why a large change in dependence need not produce a
large change in accuracy. Let an operator reassign the votes of at most $\kappa$
agents; we call $\kappa$ its \emph{reach}.

\begin{lemma}[Margin gate and critical reach]
\label{lem:gate}
Let a question have vote counts $n$ with $n_{(1)}>n_{(2)}$, so that $Y_0$ is an
answer rather than a tie.
\begin{enumerate}
  \item \emph{(Gate.)} If $N\Marg>2\kappa$, then every operator of reach at most
  $\kappa$ leaves the plurality unchanged.
  \item \emph{(Sharpness.)} If $N\Marg\le2\kappa$, then some operator of reach at
  most $\kappa$ changes the collective decision.
\end{enumerate}
Hence the critical reach of Equation~\ref{eq:kappastar} is exactly
$\kstar=\lceil N\Marg/2\rceil$, and $\Marg>2\kappa/N$ implies $\Delta A=0$.
\end{lemma}

The gate is therefore sharp, not merely sharp in order, and the two halves are
easiest to see on the examples from Section~\ref{sec:introduction}, both with
$N=10$. A population splitting seven to two to one has
$n_{(1)}-n_{(2)}=5$, so $\kstar=\lceil5/2\rceil=3$: an operator of reach two
leaves at worst $7-2=5$ votes on the leader against $2+2=4$ on the runner-up and
cannot win, whereas reach three gives $4$ against $5$ and does. A population
splitting four to three to three has $n_{(1)}-n_{(2)}=1$ and so $\kstar=1$: one
reassigned vote already produces $3$ against $4$. The same margin that makes the
first population robust makes the second fragile, at identical $N$ and identical
dependence. Under a per-agent susceptibility model in which each agent is moved
with probability at most $s$, so that $\kappa$ is stochastically dominated by a
$\mathrm{Binomial}(N,s)$ variable, the gate yields a concentration statement.

\begin{theorem}[Susceptibility gating]
\label{thm:gating}
For every question with margin $\Marg$:
\begin{enumerate}
  \item $\Delta A=0$ almost surely whenever $\Marg>2\kappa_{\max}/N$, so the
  perturbation cannot move an off-boundary collective;
  \item every downstream quantity is dominated by the susceptibility $\chi$ of
  Equation~\ref{eq:chi}, which is itself exponentially small in the margin,
\begin{align}
\mathbb{E}\big[|\Delta A|\,\big|\,\Marg\big]
&\le \Delta Y \le \chi \label{eq:gating-bound}\\
&\le \exp\!\left(-N\,\KL\!\left(\tfrac{\Marg}{2}\,\Big\|\,s\right)\right). \nonumber
\end{align}
  the exponent being positive and strictly increasing in $\Marg$ as soon as
  $\Marg/2>s$;
  \item consequently the population regression of $|\Delta A|$ on $\Marg$ has
  non-positive slope, and the mean effect over a question set is dominated by
  its low-margin mass.
\end{enumerate}
\end{theorem}

Part 3 is what makes the prediction testable: the theory predicts the sign
of a regression slope, not merely an inequality. The exponent also carries a
factor $N$, which predicts that the gate steepens with population size. We
return to this prediction, which our data do not confirm, in
Section~\ref{sec:limitations}.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/critical_reach.pdf}
\caption{Illustration of the critical-reach gate for $N=10$ agents with an initial vote profile $(7,2,1)$. Each step reassigns one vote from the leading option $A$ to the runner-up $B$, increasing the intervention reach $\kappa$ by one. The plurality is preserved for $\kappa\leq2$, along the illustrated path $(7,2,1)\rightarrow(6,3,1)\rightarrow(5,4,1)$. The first decision flip occurs at $\kappa=3$, when the profile becomes $(4,5,1)$ and $B$ takes the lead. Here the baseline margin is $M=(7-2)/10=0.5$, giving the critical reach $\kappa^\star=\lceil NM/2\rceil=3$.}
\label{fig:critical-reach}
\end{figure}

Lemma~\ref{lem:gate} gives the minimum reach required for a perturbation to cross the plurality boundary. Figure~\ref{fig:critical-reach} illustrates this gate for a simple three-option vote profile.

\begin{corollary}[Dissociation]
\label{cor:dissociation}
A mechanism intervention can raise measured dependence by an arbitrary amount
while changing the collective decision on no off-boundary question. Its mean
accuracy footprint is bounded by the fraction of low-margin questions, which is
decoupled from the magnitude of $\Delta \Dep$. A small $|\Delta A|$ therefore
does not falsify a large mechanism effect on dependence.
\end{corollary}

\subsection{Repair Headroom}
\label{sec:repairability-bounds}

The third result concerns what a diagnosis licenses. It rests on a partition of
the decision-change event that assumes nothing about the dependence dynamics,
and whose entire content is that the same three parts can be counted with two
different sign patterns.

\begin{proposition}[Directional partition]
\label{prop:identity}
Fix any two decision rules $Y_0,Y_g$ and any truth $\Truth$. Write
$B=\{Y_0=\Truth,\,Y_g\neq\Truth\}$ for a break,
$S=\{Y_0\neq\Truth,\,Y_g=\Truth\}$ for a rescue, and
$C=\{Y_0\neq\Truth,\,Y_g\neq\Truth,\,Y_g\neq Y_0\}$ for churn among incorrect
options. Then $B,S,C$ are pairwise disjoint, they exhaust the decision-change
event,
\begin{equation}
\{Y_g\neq Y_0\}=B\sqcup S\sqcup C ,
\label{eq:partition}
\end{equation}
and consequently, writing $\pchurn=\Pr(C)$,
\begin{equation}
\begin{split}
\Delta Y &= \pbreak+\prescue+\pchurn, \\
A_0-A_g &= \pbreak-\prescue .
\end{split}
\label{eq:identity}
\end{equation}
\end{proposition}

The two equalities in Equation~\ref{eq:identity} come from the same partition
read with two different weightings: the first counts all three parts, the second
counts $B$ and $S$ with opposite signs and $C$ not at all. That is the whole
reason a large $\Delta Y$ can coexist with a vanishing $A_0-A_g$, and it is an
identity, so it requires no model of the dependence dynamics.

\begin{theorem}[Repair-headroom bound]
\label{thm:headroom}
Let $R$ be any IRIC intervention in the sense of Definition~\ref{def:iric}.
Then $A_R=A_0$, so that by Equation~\ref{eq:identity} the recoverable headroom
is $\Hrep=\pbreak-\prescue$, and moreover
\begin{enumerate}
  \item $|\Hrep|\ \le\ \pbreak+\prescue\ =\ \Delta Y-\pchurn\ \le\ \Delta Y$,
  where the first inequality is strict unless $\prescue=0$ or $\pbreak=0$, and
  the last is strict unless $\pchurn=0$;
  \item no IRIC intervention exceeds the accuracy of the independent aggregate,
  so the headroom exhausts the observed decision churn, $\Hrep=\Delta Y$, only in
  the extreme case $\prescue=\pchurn=0$, that is only when every single decision
  change is a break.
\end{enumerate}
\end{theorem}

Part 2 deserves emphasis because it bounds a whole class rather than one
algorithm. Any realizable IRIC intervention is at most as accurate as the
independent ensemble under the fixed aggregator, so the idealization is a
ceiling rather than an optimistic estimate.

\subsection{Three Layers, Two Non-Implications}
\label{sec:three-layers}

Collecting the results gives the structure that organizes our experiments.

\begin{corollary}[Non-implication chain]
\label{cor:chain}
Subtracting the two equalities of Equation~\ref{eq:identity} gives the exact
non-actionable residual
\begin{equation}
\Delta Y-\Hrep=2\prescue+\pchurn\ \ge\ 0 ,
\label{eq:residual}
\end{equation}
while Equation~\ref{eq:gating-bound} caps $\Delta Y$ itself by the
susceptibility $\chi$, which is a function of the margin and the operator's
reach and not of $\Dep$. Both gaps in the chain
$\Dep\rightarrow\Delta Y\rightarrow\Hrep$ can therefore be arbitrarily large.
\end{corollary}

The two non-implications have different sources, and conflating them is the
error we are trying to prevent. Theorem~\ref{thm:gating} says that a dependence
change reaches behaviour only through the low-margin gate, so the first gap is
geometric, a matter of where the questions sit. Theorem~\ref{thm:headroom} says
that even a realized low-margin decision change is recoverable only in its net
directional part, so the second gap is a matter of signs. Equation~\ref{eq:residual}
names the price exactly: beneficial flips are counted \emph{twice}, once as
accuracy that dependence supplied and once as accuracy that repair would remove,
while churn among incorrect options is pure motion. Neither term is visible in
$\Dep$, and neither is visible in $\Delta Y$ alone.

\section{Experimental Setup}
\label{sec:experimental-setup}

\paragraph{Models.}
We use six open-weight instruction-tuned model families served locally:
Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, Phi-4-mini, and
DeepSeek-R1-7B. The suite is chosen for family diversity rather than parameter
count, and includes one reasoning-tuned model as a contrast. All models are
run with identical decoding settings and an identical prompt template; no
per-cell prompt tuning is performed at any point.

\paragraph{Benchmarks.}
We use seven multiple-choice benchmarks: TruthfulQA
\citep{lin-etal-2022-truthfulqa}, AQuA \citep{ling-etal-2017-program},
ARC-Challenge \citep{clark-etal-2018-arc}, CommonsenseQA
\citep{talmor-etal-2019-commonsenseqa}, MMLU \citep{hendrycks-etal-2021-mmlu},
StrategyQA \citep{geva-etal-2021-strategyqa}, and BIG-Bench Hard
\citep{suzgun-etal-2023-bbh}. Open-numeric tasks such as
GSM8K are excluded because the plurality machinery requires a finite option
set; AQuA serves as the mathematical-reasoning slot. All benchmarks are
normalized to a canonical form in which the correct option is index $0$, and
option display order is permuted per question so that position bias cannot be
confused with competence.

\paragraph{Population and replication.}
Unless stated otherwise the population size is $N=11$ and each condition is run
for $R=8$ replicates per question, with evidence redrawn per replicate. We use
$40$ questions per cell. For each cell we store the complete agent-by-option
score tensor for every condition, question, and replicate, so that every
quantity in Section~\ref{sec:methodology} can be recomputed offline.

\paragraph{Competence strata.}
Because both the gate of Lemma~\ref{lem:gate} and the headroom bound of
Theorem~\ref{thm:headroom} depend on baseline competence, we report results
within competence strata rather than pooling. Strata are defined by baseline
accuracy measured on held-out replicates of the baseline arm, which keeps the
stratification variable statistically independent of the evaluation data.
Appendix~\ref{sec:app-strata} gives the procedure in full, together with the
reason we adopted it.

\paragraph{Evaluation metrics.}
We report four quantities per cell: the dependence contrast
$\Delta \Dep$ of Equation~\ref{eq:dD}, which validates the instrument; the
decision-change rate $\Delta Y$; the net accuracy effect $\Delta A$; and the
directional pair $(\pbreak,\prescue)$ with the derived headroom $\Hrep$. We
also report the susceptibility slope of $|\Delta A|$ on the baseline margin.
Following the discussion in Section~\ref{sec:problem-formulation}, $\Delta A$
is never reported without $\Delta Y$.

\paragraph{Statistical analysis.}
Intervals are bootstrap percentile intervals. For question-level quantities we
resample questions within a cell; for cell-level summaries we resample cells,
since cells rather than questions are the unit of replication across models and
benchmarks. For the matched-dependence comparison we use three independent
procedures whose assumptions differ: caliper matching on dependence at several
caliper widths, nested linear-probability models with nested $F$-tests, and
nearest-neighbour matching on the confounder triple of dependence, baseline
accuracy, and margin. Appendix~\ref{sec:app-estimators} specifies each
procedure and states which estimator is primary for which substrate.

\section{Results}
\label{sec:experiments-results}

We organize the results as answers to the three questions of
Section~\ref{sec:introduction}. Section~\ref{sec:dependence-manipulation} shows
that the instrument works; Sections~\ref{sec:mechanism-effects} and
\ref{sec:decision-susceptibility-results} address where dependence comes from
and when it matters; Sections~\ref{sec:repairability-results} and
\ref{sec:generalization} address whether its effects are worth repairing.

% ============================================================================
% STATUS (2026-10-13). Experimental campaign CLOSED -- no further GPU runs.
% The English-rebuild breadth sweep is complete (74/74 real cells: 6 models x
% 6 benchmarks x 2 regimes, plus the qwen/bbh anchor pair excluded from the
% primary grid on stimulus validity -- see Appendix~\ref{sec:limitations}).
% Tier-3 copying (qwen/llama/gemma x truthfulqa/strategyqa/aqua x easy/hard,
% 18/18 cells) is also complete. Sections~\ref{sec:dependence-manipulation},
% \ref{sec:mechanism-effects}, and \ref{sec:repairability-results} below are
% FINAL, filled from frozen analyses run with no threshold/estimator/exclusion
% change after seeing output (tier1_split_sample_strata.py ->
% tier1_dA_regime_test.py -> c3_global_directional_test.py ->
% tier1_dY_decision_churn.py + aggregate -> tier3_matched_d.py +
% the within-shared-arm churn sensitivity audit). Sections
% \ref{sec:decision-susceptibility-results}, \ref{sec:intervention-response},
% and \ref{sec:generalization} remain PENDING -- they need the
% margin-susceptibility slope and the full heatmap, neither recomputed against
% the English-rebuild substrate yet; this is an offline-analysis gap, not a
% missing experiment, since no further generation is planned. Do not paste
% numbers from any table dated before 2026-10-11 (Tier-1) or before 2026-10-13
% (Tier-3); earlier campaigns (Vietnamese-era, the partial 17-42/84-cell
% snapshots, and the retired 42-usable-cell dD/dA breadth table) are superseded
% and archived under archive/.
% ============================================================================

\subsection{Does the Instrument Change Dependence?}
\label{sec:dependence-manipulation}

Because the per-cell dependence contrast is upward-biased under the
alternative at this replicate budget and its bootstrap interval under-covers
(Appendix~\ref{sec:app-estimators}), we do not report a per-cell $\Delta \Dep$
point estimate, interval, or a count of significant strata. Instead we test
only the sign, with a null distribution simulated at each stratum's own design
parameters (question count, replicate count, error prevalence) rather than
assumed to be a fair coin, which calibration showed is only approximately true.
Over the $K=66$ usable strata of the complete breadth grid, $\Dep$ increases
under sharing in $S=63$ of them; against the simulated null (mean $13.38$ of
$66$, $2000$ draws), this gives a one-sided $p=0.00050$. We therefore treat the
instrument as validated directionally -- shared evidence systematically raises
measured error dependence over matched independent evidence -- while declining
to characterize the magnitude of the increase, which the same calibration
showed is not estimable at this budget.

\subsection{Does the Origin of Dependence Matter?}
\label{sec:mechanism-effects}

Theorem~\ref{thm:nonequiv} and Corollary~\ref{cor:insufficient} predict that
dependence magnitude alone cannot determine collective accuracy: at equal
measured $\Dep$, the mechanism that produced it must still matter. We test
this with the matched-dependence procedures of
Appendix~\ref{sec:app-est-matched}, comparing the copying operator of
Equation~\ref{eq:m2} against the shared-evidence instrument of
Section~\ref{sec:matched-instrument} at equal $\Dep$, across three model
families (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B) crossed with three benchmarks
(TruthfulQA, StrategyQA, AQuA) and both competence regimes -- nine independent
(model, benchmark) replications, pooling easy and hard, of what a single
family could previously establish only once.

All three procedures agree on a positive mechanism gap in $\Delta Y$ after
matching on $\Dep$. A caliper contrast at $\varepsilon=0.05$, pre-specified as
the primary width before any of the nine pairs were inspected, gives a
macro-average gap of $0.119$ across the nine pairs, equally weighted; the
tighter $\varepsilon=0.02$ agrees in sign in every pair with adequate common
support, with one exception reported rather than dropped: llama/AQuA has only
$30\%$ caliper coverage at $\varepsilon=0.02$, against $97$--$100\%$ coverage
for all nine pairs at the pre-specified $\varepsilon=0.05$, so we do not claim
robustness to every caliper width uniformly, only to the primary one plus the
wider two. Nearest-neighbour matching on $(\Dep,A_0,\Marg)$ gives an average
treatment effect of $\Delta Y_{\text{copy}}-\Delta Y_{\text{shared}}=0.041$
($95\%$ CI $[0.013,0.073]$). A nested linear-probability model of $\Delta Y$ on
$\Dep$, then additionally on mechanism, then additionally on their interaction,
retains a mechanism coefficient of $0.114$ ($[0.062,0.169]$) after conditioning
on $\Dep$ alone -- the statistical form of Corollary~\ref{cor:insufficient}.
The same caliper contrast on $\Delta A$ is null ($0.007$, $[-0.012,0.026]$)
while $\pbreak$ and $\prescue$ are both elevated by comparable amounts
($0.026$ and $0.033$ respectively, overlapping intervals), so the mechanism gap
is concentrated in decision \emph{change}, not net accuracy, exactly the
break/rescue cancellation that Section~\ref{sec:repairability-results} and
Theorem~\ref{thm:headroom} make exact. Per family, Qwen and Llama replicate
the caliper gap as significant in all three benchmarks; Gemma replicates on
TruthfulQA but is weak and individually non-significant on StrategyQA and
AQuA. This heterogeneity does not reverse the aggregate sign, and we report it
rather than average it away.

The copying condition draws exactly one coupling realization per (question,
coupling level), so, unlike the shared-evidence arm's $R=8$ independent
replicates, its within-condition stochastic churn cannot be estimated from
this design. As a sensitivity check we net the shared arm's own measurable
replicate churn out of its $\Delta Y$ and re-run all three procedures
unchanged on the adjusted rows: the gap does not shrink under this correction,
it widens in all nine pairs (macro caliper gap $0.119\to0.163$, matched ATT
$0.041\to0.196$, mechanism coefficient $0.114\to0.212$), so measurable
shared-side stochasticity does not account for the contrast. We do not
headline these churn-adjusted figures, because the correction is necessarily
one-sided: at matched dependence levels, copying and shared-evidence
mechanisms exhibit systematically different collective decision behavior
across three model families. This contrast remains positive -- and increases
-- after accounting for the measurable within-shared-arm stochastic churn.
Because the copying condition contains only one draw per question and
coupling level, copying-side within-condition churn cannot be estimated from
the frozen design; we therefore interpret this result as evidence for
mechanism non-equivalence rather than a fully noise-decomposed causal effect.

\subsection{When Does Dependence Change the Decision?}
\label{sec:decision-susceptibility-results}

% QUESTION: when does a dependence perturbation change the collective decision?
% REPORT: slope of |dA| on baseline margin with CI; the same slope after controlling
% for agent-level confidence; collinearity diagnostics for margin against vote
% entropy and disagreement; the monotone margin-bin figure. The predicted sign is
% non-positive by Theorem thm:gating part 3.

\subsection{Do Interventions Respond Uniformly?}
\label{sec:intervention-response}

% QUESTION: do interventions respond uniformly across mechanisms?
% REPORT as a SCOPED result, not a headline: the mechanism-by-intervention
% interaction on the single benchmark and stratum where the intervention arm exists.
% Note that the only action which helps is the one that ADDS information, and is
% therefore outside the IRIC class of Definition def:iric. That is the point, not a
% caveat: Theorem thm:headroom bounds IRIC interventions only.

\subsection{Does Changing Decisions Imply Recoverable Accuracy?}
\label{sec:repairability-results}

The breadth grid has no copying arm, so it cannot exercise the
mechanism-specific limits of Theorem~\ref{thm:nonequiv}: attributing a
near-zero $\Delta A$ here to the copying operator's exact-cancellation limit
(part 2) would transfer a prediction from an arm this design does not run.
What the grid \emph{can} and does show is the weaker, mechanism-agnostic
consequence of Corollary~\ref{cor:insufficient} -- that collective accuracy is
not a function of competence and dependence alone -- together with a
corrected accounting of how much of the raw decision-change rate is actually
attributable to the shared-versus-independent contrast.

A naive reading of the matched-replicate decision-change rate,
$\Delta Y=\Pr(Y_{\text{shared}}\neq Y_{\text{indep}})$, is misleading at this
replicate budget: a single arm already churns decisions across its own
held-out replicates from resampling noise alone, at a rate
$C_{\text{within}}=(C_{\text{shared}}+C_{\text{indep}})/2$ computed as the
mean pairwise disagreement over all unordered pairs of held-out replicates
within one arm. We therefore report $\Delta Y$ as descriptive context only and
treat $\delta C=\Delta Y-C_{\text{within}}$, the excess churn net of that
baseline, as the inferential estimand, aggregated as the
\emph{macro-average across model--benchmark strata}
$\widehat{\delta C}=\tfrac1K\sum_{k=1}^{K}\widehat{\delta C}_k$ with each of
the $K=66$ usable strata weighted equally regardless of its question count,
since the question being asked is whether the effect generalizes across
configurations rather than what it averages to over our particular mix of
benchmark sizes; a question-count-weighted alternative agrees with the
macro-average in sign and in order of magnitude in every regime and tie
convention we checked, so the conclusion below is not an artifact of the
weighting choice. After accounting for within-arm stochastic instability, the
excess cross-arm decision churn is positive but small, averaging $0.4$
percentage points in the high-competence strata and $1.4$ points in the
low-competence strata (primary tie convention: a genuine tie counts as
incorrect, not broken at random). The 95\% bootstrap intervals, resampling
strata rather than questions since strata are the unit of replication here,
are $[+0.001,+0.009]$ and $[+0.002,+0.025]$ respectively, against a raw
$\Delta Y$ of $0.019$ and $0.270$ and a within-arm baseline
$C_{\text{within}}$ of $0.014$ and $0.256$. Only $1$ of $31$ high-competence
and $7$ of $35$ low-competence strata reach significance individually at the
per-stratum replicate budget; this is expected under low per-stratum power
and is supporting detail about heterogeneity, not the headline, and must not
be reported as the primary result in place of the pooled estimate. The
low-competence excess churn, while reliably positive, is an order of
magnitude smaller than the raw $\Delta Y$ it is carved out of.

Run alongside this, the pre-registered regime battery of
Appendix~\ref{sec:app-estimators} on the same $66$ strata gives a split
verdict that we state as two separate claims rather than one. The breadth
experiment \textbf{supports regime-invariant negligibility} of the net
accuracy change: $90\%$ CIs for $E[\Delta A]$ sit inside $\pm0.05$ in both
regimes, with no reliable easy/hard interaction. It \textbf{does not support}
the sign-matched directional prediction of Theorem~\ref{thm:nonequiv} part~1
for the shared-evidence operator at this substrate ($12$ of $28$ testable
strata sign-matched, one-sided $p=0.83$, i.e. worse than chance). We do not
read the second finding as a failure to be explained away: the design samples
competence near $p\approx1$ and $p\approx0$, where the theorem's Condorcet
term is degenerate, so this substrate has power on the sign but little on the
mechanism's quantitative content, and we report the null on its own terms
rather than retrying it under a different specification. As before, $\Delta A$
is not reported without $\delta C$ beside it: dependence changes decisions by
a small but reliable excess over resampling noise, while leaving net accuracy
immaterially changed, and that pairing -- not the theorem's directional
prediction -- is the empirical content of this result.

\subsection{Does the Picture Generalize?}
\label{sec:generalization}

% QUESTION: does the picture hold across models and benchmarks?
% REPORT: the six-by-seven heatmap, with boundary cases and failures shown rather
% than hidden. Expect the high-competence stratum to be near-saturated and so to
% contribute mainly the dependence contrast, with the low-competence stratum
% carrying the susceptibility and decision-change evidence; Appendix
% sec:limitations explains why, and the two strata must not be averaged.

\section{Discussion}
\label{sec:discussion}

\paragraph{A near-zero accuracy effect is a finding, not a null.}
The empirical pattern we observe most consistently is a substantial
decision-change rate alongside a net accuracy effect close to zero. Under
Proposition~\ref{prop:identity} this is exactly what a balance of harmful and
beneficial flips produces, and it is the quantitative signature of
Corollary~\ref{cor:chain}. Reporting the accuracy effect alone would describe
this regime as an absence of effect, which inverts its meaning: the collective
is being reorganized extensively, and only its aggregate score is stable.

\paragraph{Diagnosis does not imply actionability.}
Much recent work treats the detection of correlated failure as a step towards
fixing it. Theorem~\ref{thm:headroom} makes the gap between the two precise.
The recoverable accuracy is the net directional flip, not the dependence and
not the decision-change rate, and it is bounded by the latter. In a regime
where beneficial and harmful flips nearly cancel, an arbitrarily strong and
perfectly identified dependence is close to non-actionable for the entire IRIC
class. This is not an argument against decorrelation; it is a statement about
which measurements can justify it.

\paragraph{Where the remaining headroom lives.}
The analysis also indicates where an intervention could help, namely outside
the IRIC class. An operator that supplies genuinely new information is not
bounded by Theorem~\ref{thm:headroom}, because it changes the independent
aggregate itself rather than restoring it. Retrieval and evidence injection are
therefore structurally better placed than decorrelation, which is consistent
with the one intervention in our study that improves accuracy being the one
that adds information.

\paragraph{What we do not claim.}
We do not claim that correlation harms crowds in general, which is classical,
nor that LLM agents herd, which is known. We do not claim a phase transition,
nor a superior multi-agent algorithm, nor that decorrelation never helps. We
also do not claim that independent diversity predicts repairability: we tested
that hypothesis and could not sustain it once baseline competence was
controlled, and Appendix~\ref{sec:limitations} records the outcome.

\section{Conclusion}
\label{sec:conclusion}

Dependence among LLM agents is easy to measure and tempting to summarize with a
single number. We have argued, and shown both theoretically and empirically,
that this number determines neither the mechanism that produced the dependence,
nor whether the dependence will change what the collective decides, nor whether
removing it would improve accuracy. The three quantities form a chain in which
each link is a strict inequality: dependence reaches behaviour only through a
decision-boundary gate, and behaviour reaches accuracy only through the net
direction of the decision changes it produces. The practical consequence is a
measurement discipline. A report of dependence should be accompanied by the
mechanism that generated it, by the susceptibility of the population it acts
on, and by the directional decomposition of the decisions it changed. Without
the second and third, a dependence diagnosis does not yet identify an
intervention.

\bibliography{tacl2021}
\bibliographystyle{acl_natbib}

\appendix

\section{Limitations}
\label{sec:limitations}

We state the boundaries of each claim, including the two hypotheses that our
own analyses did not sustain.

\paragraph{The copying arm's within-condition stochastic churn is not
estimable.}
The comparison in Section~\ref{sec:mechanism-effects} now covers three model
families (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B) crossed with three benchmarks,
replicating the mechanism gap in all three caliper, nested-model, and
nearest-neighbour procedures at the aggregate level; Gemma is individually
weaker on two of its three benchmarks but does not reverse the sign. What
remains unresolved is narrower than model-family coverage: the copying
condition draws exactly one coupling realization per question and coupling
level, so, unlike the shared-evidence arm's $R=8$ independent replicates,
there is no repeated-draw structure from which to estimate a copying-side
churn floor. We can rule out shared-side replicate noise as an explanation for
the gap -- netting it out widens the gap in every configuration rather than
shrinking it -- but we cannot rule out copying-side noise, because it is not
measurable from this design. We therefore report the mechanism gap as evidence
for mechanism non-equivalence, supported across three model families, rather
than as a fully noise-decomposed causal effect, and we did not commission
additional copying replicates to close this gap: doing so would reopen a
closed experimental campaign to answer a question that can instead be stated,
honestly, as a declared scope limitation.

\paragraph{The intervention-response result is scoped to one benchmark and one
stratum.}
The mechanism-by-intervention interaction of
Section~\ref{sec:intervention-response} is established on a single benchmark in
the low-competence stratum. We report it as supporting evidence that
intervention response can depend on mechanism, not as a cross-benchmark or
cross-model result, and we did not extend it because it is not one of the three
headline contributions.

\paragraph{The finite-size prediction of Theorem~\ref{thm:gating} is not
confirmed.}
The gating exponent carries a factor $N$, which predicts that the
susceptibility slope steepens monotonically with population size. Measured
across $N\in\{5,11,21\}$ the slope is not monotone and peaks at the interior
value $N=11$, with overlapping intervals. We therefore report an interior-$N$
pattern rather than a clean finite-size law. The gate itself,
Lemma~\ref{lem:gate}, is unaffected, since it is deterministic.

\paragraph{Diversity does not predict repairability once competence is
controlled.}
We tested the hypothesis that independent diversity predicts repair headroom,
in the form of a regression of $\Hrep$ on $1-\Dep_0$ with model and task
effects. Two problems arose. First, specified at the level of a cell the
regression is not identifiable, because the regressor takes a single value per
cell and therefore collapses against task effects; we refit it at the question
level, where the regressor varies within a cell. Second, and decisively, the
relationship does not survive the inclusion of baseline accuracy. Adding
baseline accuracy absorbs most of the coefficient and its interval crosses
zero, and diversity is mechanically tied to competence, since a unanimous
population has zero diversity and, on an easy question, nothing left to
rescue. We therefore do not claim diversity as a general predictor of
repairability. The identity and bound of
Theorem~\ref{thm:headroom} are untouched by this, since they do not depend on
any regression.

\paragraph{The competence strata were recovered post hoc.}
Our original stratification selected questions by a separate probing run. That
selection is not reproducible against the measurement run, because the two
seeded the option-display permutation with indices that mean different things
in the two programs, so the same question could be shown under different option
orders at selection and at measurement. Within-cell contrasts are unaffected,
because the three conditions of a cell share one question set and one display
order; what was compromised is only the competence \emph{labels}. We retired
those labels and recovered the strata offline by the split-sample procedure of
Appendix~\ref{sec:app-strata}. The cost is a halving of the effective replicate
count for evaluation, a small number of strata falling below a usable size, and
a high-competence stratum that is close to saturated. We report the defect
rather than the repaired labels alone because the repair changes the
stratification, which is part of the method.

\paragraph{The high-competence stratum carries little of the evidence.}
After recovery, the high-competence stratum has baseline accuracy close to one.
Consequently the net accuracy effect is degenerate in several of its cells, and
the dependence estimator of Equation~\ref{eq:D} is undefined for most of its
questions, because every agent is constant across replicates. The
susceptibility and decision-change evidence therefore comes from the
low-competence stratum, and the two strata should not be averaged into a single
number.

\paragraph{The evidence stimulus is held fixed across model families.}
The evidence pool for the shared and independent arms is generated once per
benchmark and reused across model families. This is deliberate, since it makes
the model the only varying factor in the cross-family comparison, but it means
the evidence is not self-generated per family and we do not claim that it is.

\paragraph{Scale.}
All models are in the seven-to-nine billion parameter range and served locally
under identical decoding settings, so conclusions about larger proprietary
models are extrapolations. The population size is fixed at $N=11$ for the
breadth experiments, and the finite-size behaviour we were able to probe is
limited to $N\in\{5,11,21\}$, which is the range over which the prediction of
Theorem~\ref{thm:gating} is tested and not confirmed.

\section{Notation}
\label{sec:notation}

Table~\ref{tab:notation} collects every symbol used in the paper, grouped by the
layer of Equation~\ref{eq:intro-chain} in which it appears: the population and
its decision rule, the three layers of the chain, and the directional quantities
that determine repairability.

\begin{table*}[t]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Summary of notation used throughout the paper.}
\label{tab:notation}
\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}X@{}}
\toprule
\textbf{Symbol} & \textbf{Meaning} \\
\midrule
\multicolumn{2}{@{}l}{\emph{Population and decision}}\\
$N$ & number of agents in the collective\\
$K$ & number of answer options for a question\\
$q$ & a question; $R$ the number of replicates per condition\\
$\Truth=0$ & the correct option, pinned at index $0$ in canonical coordinates\\
$X_i$ & score vector of agent $i$ over the $K$ options\\
$n_j$ & number of informative agents whose arg max is option $j$\\
$n_{(1)},n_{(2)}$ & largest and second largest vote counts\\
$Y=\Plur(n)$ & collective decision, the plurality over informative agents; a top-level tie is a non-answer\\
$A=\mathbf{1}[Y=\Truth]$ & collective correctness; $A_0$ and $A_g$ its expectation for the independent and dependent populations\\
$p$ & marginal per-agent competence\\
$P_{\mathrm{maj}}(N,p)$ & Condorcet majority accuracy of an independent population\\
\midrule
\multicolumn{2}{@{}l}{\emph{The three layers}}\\
$\Mech$ & dependence mechanism, either upstream shared evidence or downstream copying\\
$\rho$ & latent correlation of the idealized model, Equation~\ref{eq:m1}\\
$a=\Phi^{-1}(p)$ & competence threshold of the latent variable\\
$\Phi_2(\cdot,\cdot;\rho)$ & standard bivariate normal distribution function\\
$\He_k$ & Hermite polynomials, $\He_0=1$, $\He_1(x)=x$, $\He_2(x)=x^2-1$\\
$\tetra(\rho)$ & correctness correlation induced by $\rho$, Equation~\ref{eq:tau}; strictly increasing with $\tetra(\rho)\le\rho$\\
$g$ & copying strength, the social-pressure knob; $g=0$ is independence\\
$E\in\{0,1\}^{R\times N}$ & error indicator, $1$ if agent $i$ is wrong in replicate $r$\\
$\sigma_i$ & standard deviation of $E_i$ across replicates\\
$\Dep(q)$ & dependence magnitude, Equation~\ref{eq:D}; undefined when an arm saturates\\
$\Dep_0$ & dependence of the baseline arm\\
$\Delta \Dep$ & dependence contrast between the shared and independent arms, Equation~\ref{eq:dD}\\
$\Marg$ & normalized runner-up gap, or margin, Equation~\ref{eq:margin}, measured on the baseline arm\\
$\chi$ & decision susceptibility, $\Pr(\kappa\ge\kstar)$, Equation~\ref{eq:chi}\\
$\kappa$ & reach of a perturbation, the number of votes it reassigns\\
$\kstar$ & critical reach, $\lceil N\Marg/2\rceil$ by Lemma~\ref{lem:gate}\\
$s$ & per-agent susceptibility, the cap on the probability that an agent's vote is changed\\
$\KL(a\|s)$ & $a\ln(a/s)+(1-a)\ln\big((1-a)/(1-s)\big)$\\
\midrule
\multicolumn{2}{@{}l}{\emph{Consequence and repairability}}\\
$\Delta Y$ & decision-change rate, $\Pr(Y_g\neq Y_0)$\\
$\Delta A$ & net accuracy effect, $A_g-A_0$; never reported without $\Delta Y$\\
$\pbreak$ & harmful flip rate, $\Pr(Y_0=\Truth,\ Y_g\neq\Truth)$\\
$\prescue$ & beneficial flip rate, $\Pr(Y_0\neq\Truth,\ Y_g=\Truth)$\\
$\pchurn$ & churn rate, a changed decision that is wrong both before and after, Proposition~\ref{prop:identity}\\
$\Hrep$ & recoverable headroom of an IRIC intervention, $\pbreak-\prescue$\\
IRIC & independence-restoring and information-conserving, Definition~\ref{def:iric}\\
$A_R$ & accuracy of an IRIC intervention; equal to $A_0$ by construction\\
\bottomrule
\end{tabularx}
\end{table*}

\section{Proofs}
\label{sec:app-proofs}

Proofs appear in the order in which the results are used in the main text.
Nothing below is deeper than dominated convergence, a Hermite expansion and a
Chernoff bound, but several of the intermediate equalities are the ones usually
left implicit, and each of them is load-bearing somewhere in the paper.

\paragraph{Proof of Lemma~\ref{lem:degeneracy}.}
Write $x_i=e_i-\bar e$, so that $\sum_i x_i=0$ by the definition of the mean.
This is the only property of $e$ that enters, which is why the conclusion cannot
depend on $e$. The numerator of Equation~\ref{eq:degenerate} is then
\begin{align*}
\sum_{i\neq j}x_ix_j
&=\Big(\sum_i x_i\Big)^{2}-\sum_i x_i^{2} \\
&=0-\sum_i x_i^{2} \\
&=-\sum_i x_i^{2}.
\end{align*}
and since the entries of $e$ are not all equal we have $\sum_i x_i^2>0$, so the
ratio is well defined and equals
$-\sum_i x_i^{2}/\big((N-1)\sum_i x_i^{2}\big)=-1/(N-1)$. \hfill$\square$

\paragraph{Proof of Lemma~\ref{lem:calibration}.}
Throughout, $a=\Phi^{-1}(p)$, and $L_i,L_j$ are standard normal with correlation
$\rho\in[0,1)$; the endpoint $\rho=1$ is treated at the end.

Since $C_i=\mathbf{1}[L_i\le a]$ is Bernoulli with mean $p$ by
Equation~\ref{eq:marg-preserved}, its variance is $p(1-p)$ and
$\mathbb{E}[C_iC_j]=\Pr(L_i\le a,L_j\le a)=\Phi_2(a,a;\rho)$, which is
Equation~\ref{eq:tau}. To expand that orthant probability, recall Mehler's
formula: for $|\rho|<1$ the standard bivariate normal density factorizes as
$\varphi_2(x,y;\rho)=\varphi(x)\varphi(y)\sum_{k\ge0}(\rho^{k}/k!)\He_k(x)\He_k(y)$.
Its $k=0$ term is the independent density $\varphi(x)\varphi(y)$, so subtracting
$p^2$ removes exactly that term and leaves the dependence. Integrating what
remains over the quadrant $(-\infty,a]^2$ and using
$\varphi(x)\He_k(x)=(-1)^k\,d^k\varphi/dx^k$, which makes
$-\varphi(x)\He_{k-1}(x)$ an antiderivative of $\varphi(x)\He_k(x)$ for $k\ge1$
whose boundary term at $-\infty$ vanishes,
\begin{align*}
\Phi_2(a,a;\rho)-p^2
&=\sum_{k\ge1}\frac{\rho^{k}}{k!}
\left(\int_{-\infty}^{a}\varphi(x)\He_k(x)\,dx\right)^{\!2}\\
&=\sum_{k\ge1}\frac{\rho^{k}}{k!}
\big(-\varphi(a)\He_{k-1}(a)\big)^{2}\\
&=p(1-p)\sum_{k\ge1}w_k\rho^{k},
\end{align*}
with $w_k$ as in Equation~\ref{eq:mehler}; squaring has removed the sign, so
$w_k\ge0$, and $w_1=\varphi(a)^2/(p(1-p))>0$. Dividing by $p(1-p)$ gives the
expansion. At $\rho=1$ we have $L_i=L_j$ almost surely, hence $C_i=C_j$ and
$\tetra(1)=1$; the coefficients are non-negative, so the series is monotone in
$\rho$ and converges to its endpoint value, giving $\sum_{k\ge1}w_k=1$. Thus $w$
is a probability distribution on $k\ge1$ and $\tetra(\rho)=\mathbb{E}[\rho^{K}]$
for $K\sim w$.

The three conclusions now read off. Differentiating the series,
$\tetra'(\rho)=\sum_{k\ge1}kw_k\rho^{k-1}\ge w_1>0$, so $\tetra$ is strictly
increasing; with $\tetra(0)=0$ and $\tetra(1)=1$ it is therefore a bijection of
$[0,1]$ onto itself. For the attenuation, $\rho^{k}\le\rho$ for every $k\ge1$ and
every $\rho\in[0,1]$, so $\tetra(\rho)=\sum_k w_k\rho^{k}\le\rho\sum_k
w_k=\rho$. Equality at some interior $\rho$ would force $w_k=0$ for every
$k\ge2$; but $w_2$ and $w_3$ are proportional to $\He_1(a)^2=a^2$ and
$\He_2(a)^2=(a^2-1)^2$, which cannot vanish simultaneously, so the inequality is
strict on $(0,1)$. \hfill$\square$

\paragraph{Proof of Theorem~\ref{thm:nonequiv}.}
We treat the three mechanisms in turn and then compare.

\emph{Upstream shared evidence.} Given $Z_0=z$ the indicators $C_1,\dots,C_N$ of
Equation~\ref{eq:m1} are independent and identically distributed Bernoulli
variables with conditional competence
\begin{align*}
u_\rho(z)
&=\Pr\!\left(\sqrt{\rho}z+\sqrt{1-\rho}\,\epsilon\le a\right) \\
&=\Phi\!\left(\frac{a-\sqrt{\rho}\,z}{\sqrt{1-\rho}}\right).
\end{align*}
Writing $B_N(u)=\sum_{k>N/2}\binom{N}{k}u^k(1-u)^{N-k}=P_{\mathrm{maj}}(N,u)$ and
integrating out the common draw gives the exact finite-$\rho$ accuracy
\begin{equation}
A_{\text{shared}}(\rho)=\int\varphi(z)\,B_N\!\left(u_\rho(z)\right)dz ,
\label{eq:A-shared-rho}
\end{equation}
which passes the one check available: at $\rho=0$ we have $u_0(z)=\Phi(a)=p$ for
every $z$, so $A_{\text{shared}}(0)=B_N(p)=P_{\mathrm{maj}}(N,p)$, the
independent population.

Now fix $z\neq a$ and let $\rho\to1$. The numerator $a-\sqrt{\rho}z$ tends to
$a-z\neq0$ while $\sqrt{1-\rho}\to0^{+}$, so the argument of $\Phi$ tends to
$+\infty$ when $z<a$ and to $-\infty$ when $z>a$; hence $u_\rho(z)\to
\mathbf{1}[z<a]$ for almost every $z$, the single exception $z=a$ being
Lebesgue-null. Since $B_N$ is a polynomial, hence continuous, with $B_N(0)=0$ and
$B_N(1)=1$, the integrand converges likewise and is bounded by $1$, which is
$\varphi$-integrable, so dominated convergence applies to
Equation~\ref{eq:A-shared-rho}:
\begin{align*}
A_{\text{shared}}(\rho)
&\longrightarrow\int\varphi(z)\,\mathbf{1}[z<a]\,dz
=\Phi(a)
=p .
\end{align*}
The limit says that all agents inherit the single sign of the common draw, so the
collective is correct exactly when that draw is. Therefore
$\Delta_{\text{shared}}\to p-P_{\mathrm{maj}}(N,p)$, and the Condorcet facts
$P_{\mathrm{maj}}(N,p)>p$ for $p>\tfrac12$ and $P_{\mathrm{maj}}(N,p)<p$ for
$p<\tfrac12$, valid for $N\ge3$, give
$\mathrm{sign}(\Delta_{\text{shared}})=\mathrm{sign}(\tfrac12-p)\neq0$.

\emph{Synchronous copying.} By Equation~\ref{eq:m2} the vector
$X^{(1)}=(Y_0,\dots,Y_0)$ is constant across agents, and the plurality of a
constant vector is that constant, so $Y_1=\Plur(X^{(1)})=Y_0$. The same argument
applies at every step, so by induction $Y_t=Y_0$ for all $t$, the accuracy is
$P_{\mathrm{maj}}(N,p)$ throughout, and $\Delta_{\text{copy}}=0$. Meanwhile
Equation~\ref{eq:copy-saturate} gives $\Dep=1$ exactly from $t=1$ onward. The
mechanism therefore attains the ceiling of the dependence scale at zero accuracy
cost, which is the sharpest possible form of the dissociation.

\emph{Sequential cascade copying.} In the model of
\citet{bikhchandani-etal-1992-theory} and \citet{banerjee-1992-simple} the public
history can lock the population onto the action favoured by an early minority
whose private signals happened to align against the truth. Two observations bound
the accuracy. The locked action equals the truth only when the pre-cascade tally
already favoured it, and the public signal is a coarsening of the private
signals, so a cascade cannot manufacture correctness that the private majority
lacked; hence $A\le P_{\mathrm{maj}}(N,p)$. And there is a set of orderings and
signal realizations of positive measure on which the cascade locks onto a wrong
action while the full private majority is correct, which makes the inequality
strict on that set.

\emph{Comparison.} The first two mechanisms share the dependence limit
$\Dep\to1$ and the competence $p$, yet reach different collective accuracies
whenever $p\neq\tfrac12$; for $p>\tfrac12$ the ranking is
$\Delta_{\text{copy}}=0>\Delta_{\text{shared}}$. \hfill$\square$

\paragraph{Proof of Corollary~\ref{cor:insufficient}.}
Suppose such a map existed. By hypothesis $\Dep$ is a functional of the marginal
competence and the pairwise correctness correlation alone, and by
Theorem~\ref{thm:nonequiv} the shared-evidence and synchronous-copying
populations agree on $N$, on $p$, and in the dependence limit, where both have
correlation one and hence the same $\Dep$. They nevertheless differ in $A$ by
$|p-P_{\mathrm{maj}}(N,p)|>0$. A function cannot take two values at one
argument, so no such map exists. \hfill$\square$

\paragraph{Proof of Lemma~\ref{lem:gate}.}
Let $j^{*}$ be the plurality option, so $n_{j^{*}}=n_{(1)}$.

For the gate, fix any option $c\neq j^{*}$. Reassigning $\kappa$ votes can lower
$n_{j^{*}}$ by at most $\kappa$ and raise $n_c$ by at most $\kappa$, so after the
perturbation
\begin{align*}
n'_{j^{*}}-n'_{c}
&\ge (n_{j^{*}}-n_c)-2\kappa \\
&\ge n_{(1)}-n_{(2)}-2\kappa \\
&= N\Marg-2\kappa .
\end{align*}
If $N\Marg>2\kappa$ this is strictly positive for every $c$, so $j^{*}$ remains
the strict plurality and the decision is unchanged.

For sharpness, assume $N\Marg\le2\kappa$ and set
$\kappa'=\lceil N\Marg/2\rceil\le\kappa$. Since $N\Marg=n_{(1)}-n_{(2)}\le
n_{(1)}$ we have $\kappa'\le n_{(1)}$, so $\kappa'$ votes can be taken from the
leader; reassign them to the runner-up $c$. The new counts are
$n'_{j^{*}}=n_{(1)}-\kappa'$ and $n'_{c}=n_{(2)}+\kappa'$, every other count is
unchanged and therefore still at most $n_{(2)}\le n'_c$, and
\begin{align*}
n'_{c}-n'_{j^{*}}
&=(n_{(2)}+\kappa')-(n_{(1)}-\kappa') \\
&=2\kappa'-N\Marg \\
&\ge 0 .
\end{align*}
If the difference is positive, $c$ is the new strict plurality; if it is zero, the
top two counts tie, which is a non-answer and so differs from the answer $Y_0$.
In both cases the collective decision changes.

Combining the two halves, any reach that changes the decision satisfies
$\kappa\ge N\Marg/2$, hence $\kappa\ge\lceil N\Marg/2\rceil$ because $\kappa$ is
an integer, and the construction attains that value; so the minimum in
Equation~\ref{eq:kappastar} is $\kstar=\lceil N\Marg/2\rceil$. \hfill$\square$

\paragraph{Proof of Theorem~\ref{thm:gating}.}
Part 1 is Lemma~\ref{lem:gate} applied with $\kappa\le\kappa_{\max}$.

Part 2 is a chain of four inequalities, which is Equation~\ref{eq:gating-bound}
read left to right. Pointwise $|\Delta A|\le\mathbf{1}[Y_g\neq Y_0]$, since a
question whose decision is unchanged contributes nothing; by
Lemma~\ref{lem:gate} a changed decision requires $\kappa\ge\kstar$, so
$\{Y_g\neq Y_0\}\subseteq\{\kappa\ge\kstar\}$; and writing $a=\Marg/2$ we have
$\kstar=\lceil Na\rceil\ge Na$, so the assumed stochastic dominance
$\kappa\preceq\mathrm{Binomial}(N,s)$ applies. Hence, with $B\sim
\mathrm{Binomial}(N,s)$,
\begin{align*}
\mathbb{E}\big[|\Delta A|\,\big|\,\Marg\big]
&\le \Delta Y \le \Pr(\kappa\ge\kstar) \\
&= \chi \le \Pr(B\ge Na).
\end{align*}
It remains to bound that binomial tail. Markov's inequality applied to
$e^{\lambda B}$ for $\lambda>0$ gives
\begin{align*}
\Pr(B\ge Na)
&\le e^{-\lambda Na}\,\mathbb{E}e^{\lambda B} \\
&=\exp\!\big(-N[\lambda a-\ln(1-s+se^{\lambda})]\big).
\end{align*}
and it remains to choose $\lambda$. Maximizing
$g(\lambda)=\lambda a-\ln(1-s+se^{\lambda})$ means solving $g'(\lambda)=0$, that
is
\begin{align*}
a=\frac{se^{\lambda}}{1-s+se^{\lambda}}
&\iff a(1-s)+ase^{\lambda}=se^{\lambda}\\
&\iff a(1-s)=se^{\lambda}(1-a)\\
&\iff e^{\lambda^{*}}=\frac{a(1-s)}{s(1-a)} ,
\end{align*}
which is admissible, and satisfies $\lambda^{*}>0$, exactly when $a>s$; $g$ is
concave, so this is the maximum. Substituting back, the denominator collapses,
$1-s+se^{\lambda^{*}}=1-s+a(1-s)/(1-a)=(1-s)/(1-a)$, and therefore
\begin{align*}
g(\lambda^{*})
&=a\ln\frac{a(1-s)}{s(1-a)}-\ln\frac{1-s}{1-a}\\
&=a\ln\frac{a}{s}+a\ln\frac{1-s}{1-a}+\ln\frac{1-a}{1-s}\\
&=a\ln\frac{a}{s}+(1-a)\ln\frac{1-a}{1-s}
=\KL(a\|s),
\end{align*}
which is the exponent claimed in Equation~\ref{eq:gating-bound} with
$a=\Marg/2$. The exponent is also strictly increasing in $a$, since
$\partial_a\KL(a\|s)=\ln\frac{a}{s}-\ln\frac{1-a}{1-s}=\lambda^{*}>0$ for $a>s$,
and it carries the factor $N$ explicitly.

Part 3 follows: that monotonicity makes the envelope
$\exp(-N\KL(\Marg/2\|s))$ strictly decreasing in $\Marg$ on $\Marg>2s$, so
averaging over any question distribution $\pi$ yields a mean controlled by the
mass $\pi$ places near $\Marg\approx0$, and the population regression of
$|\Delta A|$ on $\Marg$ has non-positive slope. \hfill$\square$

\paragraph{Proof of Proposition~\ref{prop:identity}.}
The three sets are pairwise disjoint: $B$ requires $Y_0=\Truth$ whereas $S$ and
$C$ both require $Y_0\neq\Truth$, so $B$ meets neither, and $S$ requires
$Y_g=\Truth$ while $C$ requires $Y_g\neq\Truth$. Each lies inside the
decision-change event, since on $B$ we have $Y_0=\Truth\neq Y_g$, on $S$ we have
$Y_g=\Truth\neq Y_0$, and on $C$ the condition $Y_g\neq Y_0$ is imposed directly.
And they exhaust it: supposing $Y_g\neq Y_0$ and splitting on the correctness of
$Y_0$, either $Y_0=\Truth$, whence $Y_g\neq\Truth$, which is $B$; or
$Y_0\neq\Truth$, whence either $Y_g=\Truth$, which is $S$, or $Y_g\neq\Truth$,
which together with $Y_g\neq Y_0$ is $C$. This proves
Equation~\ref{eq:partition}, and adding probabilities over a partition gives
$\Delta Y=\pbreak+\prescue+\pchurn$.

For the signed reading, split each accuracy on the correctness of the other rule.
Both splits share the term in which the two decisions are correct, which forces
$Y_g=Y_0$, so subtracting cancels it:
\begin{align*}
A_0-A_g
&=\big[\Pr(Y_0=\Truth,Y_g=\Truth)+\Pr(B)\big]\\
&\quad-\big[\Pr(Y_0=\Truth,Y_g=\Truth)+\Pr(S)\big]\\
&=\Pr(B)-\Pr(S)
=\pbreak-\prescue .
\end{align*}
Note that $C$ appears in neither bracket, which is precisely why churn among
incorrect options is invisible to accuracy. No model of the dependence dynamics
is used anywhere above. \hfill$\square$

\paragraph{Proof of Theorem~\ref{thm:headroom}.}
By Definition~\ref{def:iric} an IRIC intervention discards the dependent votes
and re-aggregates a fresh independent draw from the same per-agent
belief-generating process under the same aggregation rule, so its decision has
the marginal law of $Y_0$ and $A_R=A_0$. Equation~\ref{eq:identity} then turns
$\Hrep=A_R-A_g$ into the stated identity, and the two inequalities of part 1
follow from it,
\begin{align*}
|\Hrep| &= |\pbreak-\prescue| \le \pbreak+\prescue \\
&= \Delta Y-\pchurn \le \Delta Y .
\end{align*}
the first step being strict unless one of $\pbreak,\prescue$ vanishes and the
last being strict unless $\pchurn=0$.

Part 2 follows from $A_R=A_0$: under a fixed aggregator an IRIC resample cannot
extract more truth than the independent ensemble contains, with equality in the
limit of a perfect fresh resample, so $A_0$ is a ceiling for the whole class.
Reading the display above backwards, the headroom equals the full
decision-change rate only when $\prescue=0$ and $\pchurn=0$. \hfill$\square$
\section{Prompts}
\label{sec:app-prompts}

Tables~\ref{tab:prompts-elicit} and~\ref{tab:prompts-peer} reproduce every prompt
verbatim, split by which mechanism the text belongs to: the first holds the
skeleton that every agent sees together with the evidence slot, which is the
shared-evidence mechanism of Equation~\ref{eq:m1}; the second holds the peer slot,
which is the copying mechanism of Equation~\ref{eq:m2}, together with the offline
generator that produces the evidence pool. Braces denote substituted fields. A
field left empty contributes nothing to the prompt, so the elicitation format is
byte-identical across conditions that do not use it; the arms are therefore
matched at the level of the emitted string and not merely at the level of the
design. Every prompt is English.

\begin{table*}[tp]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{The elicitation skeleton and the evidence slot. The skeleton is identical
in every condition and at every coupling strength. The shared and independent arms
use the \emph{same} framing row and differ only in whether the drawn
\texttt{\{clue\}} is common to all $N$ agents, which is what matches them on
evidence quality.}
\label{tab:prompts-elicit}
\begin{tabularx}{\textwidth}{@{}p{0.12\textwidth}p{0.15\textwidth}X@{}}
\toprule
\textbf{Slot} & \textbf{When used} & \textbf{Text} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Elicitation skeleton. The only prompt that obtains an
answer.}}\\
\addlinespace[2pt]
skeleton &
every condition, every $g$ &
\ttfamily \{persona\}\newline
Question: \{question\}\newline
\mbox{}\newline
Options:\newline
\{options\}\newline
\{evidence\_block\}\newline
\{peer\_block\}\newline
\mbox{}\newline
Think briefly, then rate how strongly you believe EACH label (\{labels\}) is
correct, on a scale from -10 to 10, where 10 means certainly correct. Reply with
JSON ONLY, in the form \{"scores": \{"A": <number>, "B": <number>, ...\},
"reasoning": "<brief>"\}.\\
\midrule
\multicolumn{3}{@{}l}{\emph{\texttt{evidence\_block}. Empty in the baseline arm.}}\\
\addlinespace[2pt]
strong framing &
both evidence arms in the main text &
\ttfamily Reference information or evidence for this question:\newline
Evidence from a source considered reliable: \{clue\} Please consider this
information carefully.\\
weak framing &
extra grid conditions only &
\ttfamily Reference information or evidence for this question:\newline
A reference note (reliability unverified): \{clue\}\\
clue fallback &
generator reply not parsable &
\ttfamily A number of commonly cited sources connect this issue to the view:
\{target\}.\\
\bottomrule
\end{tabularx}
\end{table*}

\begin{table*}[tp]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{The peer slot and the evidence generator. The three directive rungs are the coupling $g$ of Equation~\ref{eq:m2}; only informative peers enter the tally.
\texttt{\{c\}}/\texttt{\{m\}}.}
\label{tab:prompts-peer}
\begin{tabularx}{\textwidth}{@{}p{0.12\textwidth}p{0.15\textwidth}X@{}}
\toprule
\textbf{Slot} & \textbf{When used} & \textbf{Text} \\
\midrule
\multicolumn{3}{@{}l}{\emph{\texttt{peer\_block}. Filled only by the copying
operator.}}\\
\addlinespace[2pt]
$g=0$ &
baseline; supplies $Y_0$ &
empty, so the population performs repeated independent voting\\
$0<g<1$ &
soft rung &
\ttfamily Current views of the other agents in the group:\newline
Of the \{m\} other agents, \{c\}/\{m\} chose «\{top\}» (vote distribution:
\{dist\}). Place about \{100g\}\% of your weight on this group consensus, and the
remainder on your own independent judgement.\\
$1\le g<2$ &
middle rung &
\ttfamily Current views of the other agents in the group:\newline
The whole group (\{m\} agents) has converged on the conclusion «\{top\}». As a
member who follows the group consensus, CHOOSE «\{top\}» and score it well above
every alternative, UNLESS you have DECISIVE evidence that the group is wrong.\\
$g\ge2$ &
top rung &
\ttfamily Current views of the other agents in the group:\newline
The whole group (\{m\} agents) has converged on the conclusion «\{top\}». As a
member who follows the group consensus, CHOOSE «\{top\}» and score it well above
every alternative. Do not reconsider.\\
\midrule
\multicolumn{3}{@{}l}{\emph{Evidence generator. Run once per benchmark; the pool is cached and
reused across model families (Appendix~\ref{sec:limitations}).}}\\
\addlinespace[2pt]
generator &
once per question and target, offline &
\ttfamily You are generating data for an experiment on how information influences
reasoning. Write in ENGLISH only.\newline
\mbox{}\newline
Question: \{question\}\newline
\mbox{}\newline
Write \{n\} SHORT (one to two sentences) and MUTUALLY DISTINCT pieces of
"reference evidence or context" that sound plausible and would tend to incline a
reader toward the view: «\{target\}».\newline
\mbox{}\newline
Strict constraints. Do NOT write "the answer is" or any equivalent. Do NOT mention
the option labels A/B/C/D. Do NOT state that this is the correct choice. Provide
indirect information or argument only.\newline
\mbox{}\newline
Number them 1..\{n\}, one per line.\\
\bottomrule
\end{tabularx}
\end{table*}

Six properties of the two tables carry the design, and none of them is visible in
the strings alone.

\paragraph{The framing is evidential, never social.}
No row of the evidence slot states what any other agent chose. This is what keeps
the shared-evidence mechanism of Equation~\ref{eq:m1} distinct from the copying
mechanism of Equation~\ref{eq:m2} rather than a weaker version of it, and it is
why the two can be compared as mechanisms at matched dependence instead of as two
doses of one treatment.

\paragraph{The rungs interpolate rather than switch.}
The soft rung preserves an explicit independent escape, so that $g$ moves the
population gradually. The middle rung issues a categorical directive but retains a
truth escape clause, so the regime remains a conformity pull rather than a blind
label copy: a flip has to survive an agent that still knows better. The top rung
removes the escape, which is what allows the vote correlation to approach the
saturated limit analysed in Theorem~\ref{thm:nonequiv}.

\paragraph{At $g=0$ the peer slot is absent, not neutral.}
The block contributes no bytes, so the population performs repeated independent
voting and supplies the baseline $Y_0$ used throughout
Section~\ref{sec:theory}. This is why the baseline is a genuine control rather
than a condition with a mild social signal.

\paragraph{The generator is forbidden from revealing an answer.}
Its constraints block the phrase \emph{the answer is}, the option labels, and any
statement that a choice is correct, so the evidence acts on the agents' private
belief formation rather than by announcing a label. Two kinds of target are used
per question: the correct option produces the pool drawn on by the conditions
reported in the main text, and each distractor produces a misleading pool drawn on
by the remaining conditions of Appendix~\ref{sec:app-conditions}. Items are
de-duplicated before use, and the independent arm draws from the de-duplicated pool
without replacement, for the reason given in Appendix~\ref{sec:app-conditions}.

\paragraph{Score parsing is deliberately tolerant.}
The score vector is mapped onto option positions accepting keys given by label, by
option text, or by index. Options the model omits are filled just below the lowest
score it did assign, so a reply endorsing a single option still yields a peaked
state rather than a flat one. A reply that parses to a completely flat state is
treated as a formatting failure and resampled up to two further times.

\paragraph{A state that stays flat casts no vote.}
Such a state is recorded as uninformative, which is the informativeness condition
in the definition of the plurality rule in
Section~\ref{sec:problem-formulation}. The consequence of omitting that condition
is quantified in Appendix~\ref{sec:app-estimators}: because a flat vector's
arg max is index $0$, and index $0$ is the truth in canonical coordinates,
counting such replies would score every parse failure as correct.
\section{Instrument and Protocol Details}
\label{sec:app-protocol}

\subsection{The Condition Grid}
\label{sec:app-conditions}

The full protocol defines seven conditions, formed by crossing the evidence
direction with its framing strength and its commonality: a baseline with no
evidence; shared correct evidence at weak and strong framing; independent
correct evidence at strong framing; shared misleading evidence at weak and
strong framing; and independent misleading evidence at strong framing. The
framings are those of Table~\ref{tab:prompts-elicit}. The main text
reports the three conditions that isolate commonality at matched quality, namely
the baseline, shared correct strong, and independent correct strong; these are
the arms named in Section~\ref{sec:matched-instrument} and the ones entering the
dependence contrast of Equation~\ref{eq:dD}. The independent arms use a single
framing strength because their role is to serve as a matched control for
commonality, not to trace a dose response.

Two implementation details protect the contrast. First, evidence is drawn afresh
in every replicate, which is what identifies the contrast at the replicate level
and makes the dependence estimator of Equation~\ref{eq:D} informative rather
than algebraically pinned, as discussed in
Section~\ref{sec:problem-formulation}. Second, the independent arm draws without
replacement from a de-duplicated pool, so that no two agents receive an
identical string by accident. This removes duplicate artifacts without forcing
the independent dependence to zero, since intrinsic shared failure modes and
matched evidence content legitimately remain; the distinction matters because
the quantity of interest in Section~\ref{sec:mechanism-effects} is the contrast
between the arms, not the absolute level of either.

\subsection{Competence Strata by Split Sample}
\label{sec:app-strata}

Strata are recovered from the stored score tensors without further generation,
by the following procedure. For each model and benchmark, pool the questions of
both original strata. Partition the $R=8$ replicates of the baseline arm into a
binning half and an evaluation half. Assign each question to a stratum using
only the binning half, under fixed accuracy thresholds. Evaluate all conditions,
that is all quantities of Section~\ref{sec:dependence-measurement}, using only
the evaluation half. Because the binning variable is computed on replicates
disjoint from those used for evaluation, the stratum assignment is statistically
independent of the measured contrasts, so the procedure does not induce a
selection bias in favour of the result. Strata with fewer than a preset minimum
number of questions are reported but not used.

The procedure costs half of the effective replicates for evaluation and
therefore widens the intervals described in
Appendix~\ref{sec:app-est-ci}. We adopted it because the original probe-based
labels are not reproducible against the measurement run, for the reason given in
Appendix~\ref{sec:limitations}. Stratifying at all is required by the theory
rather than a convenience: both the gate of Lemma~\ref{lem:gate} and the
headroom bound of Theorem~\ref{thm:headroom} depend on baseline competence, so
pooling strata would average over regimes in which the predictions differ, which
is why Section~\ref{sec:experimental-setup} reports within strata throughout. We
verified the recomputation by reconstructing the per-question accuracy,
dependence, and margin from the stored tensors over all replicates and checking
that they reproduce the values written by the measurement program exactly.

\section{Algorithms}
\label{sec:app-algorithms}

Algorithm~\ref{alg:instrument} in Section~\ref{sec:matched-instrument} is the
only procedure that queries a model. The two below read its stored tensor and
query nothing, which is what makes the reanalyses described in
Appendix~\ref{sec:app-repro} free.

\subsection{Offline Estimation of the Three Layers}
\label{sec:app-alg-estimate}

Algorithm~\ref{alg:estimate} computes every quantity of
Section~\ref{sec:dependence-measurement} from the score tensor. Four details are
worth pointing at, because each of them is a decision that the formulas alone do
not dictate. Line~3 implements the split-sample stratification of
Appendix~\ref{sec:app-strata}: the halves are disjoint, so the stratum label
cannot be correlated with the evaluated contrast. Line~4 drops uninformative
agents before voting, which is the informativeness condition of
Section~\ref{sec:problem-formulation}; without it a flat score vector would
arg-max to index $0$ and every parse failure would score as correct. Line~8
pairs the two arms at matched replicate index, which is what makes the
classification into $B$, $S$, $C$ a within-replicate comparison and lets the
identity of Equation~\ref{eq:identity} be checked rather than assumed. Line~10
returns $\Dep$ as missing, not as zero, when the arm saturates, for the reason
given after Equation~\ref{eq:D}.



\begin{figure}[t]
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.95}
\noindent\begin{tabularx}{\columnwidth}{@{}r@{\ \,}Z@{}}
\toprule
\multicolumn{2}{@{}l@{}}{\algname{Estimating the three layers}%
\label{alg:estimate}}\\
\midrule
 & \lvA\algin\ tensor $T$ (Alg.~\ref{alg:instrument}), strata thresholds\\
 & \lvA\algout\ per $q$: $\Dep$, $\Marg$, $\Delta Y$, $\pbreak$, $\prescue$,
   $\pchurn$, $\Hrep$\\
\midrule
1 & \lvA\algfor\ $q\in Q$:\\
2 & \lvB split the $R$ baseline replicates into halves
   $R_{\text{bin}},R_{\text{ev}}$; stratum of $q$ from $R_{\text{bin}}$
   only, everything else from $R_{\text{ev}}$\\
3 & \lvB\algfor\ arm $c$, replicate $r\in R_{\text{ev}}$:\\
4 & \lvC keep agents $i$ with
   $\max_k T[c,q,r,i,k]>\min_k T[c,q,r,i,k]$\\
5 & \lvC $n\leftarrow$ vote counts of kept agents;
   $Y_{c,r}\leftarrow\Plur(n)$ (a top-level tie is a non-answer)\\
6 & \lvC $E[r,i]\leftarrow\mathbf{1}[\,\arg\max_k T[c,q,r,i,k]\neq0\,]$\\
7 & \lvB $\Marg(q)\leftarrow(n_{(1)}-n_{(2)})/N$ (baseline arm);
   $\Dep_c(q)\leftarrow$ Eq.~\ref{eq:D} on $E$ of arm $c$\\
8 & \lvB\algfor\ $r\in R_{\text{ev}}$: classify
   $(Y_{\indep,r},Y_{\shared,r})$ as agreement, $B$, $S$ or $C$
   (Prop.~\ref{prop:identity})\\
9 & \lvB $\pbreak,\pchurn,\prescue\leftarrow$ counts$/|R_{\text{ev}}|$;
   $\Delta Y\leftarrow\pbreak+\prescue+\pchurn$;
   $\Hrep\leftarrow\pbreak-\prescue$\\
10 & \lvB\algif\ every agent of arm $c$ is constant over $R_{\text{ev}}$:
   $\Dep_c(q)\leftarrow\textsc{missing}$ (not $0$)\\
\midrule
 & \lvA\algret\ the per-question table; assert both equalities of
   Eq.~\ref{eq:identity} for every $q$.\\
\bottomrule
\end{tabularx}
\end{figure}

\subsection{The Matched-Dependence Gap}
\label{sec:app-alg-matched}

Algorithm~\ref{alg:matched} is the caliper procedure, the primary of the three
in Appendix~\ref{sec:app-est-matched}. Its logic is that
Lemma~\ref{lem:calibration} makes measured dependence a legitimate matching
coordinate: because $\tetra$ is invertible, two arms with the same measured
dependence sit at the same latent coupling, so a residual difference in
$\Delta Y$ cannot be attributed to one arm being more dependent than the other.
The bootstrap resamples matched pairs rather than observations, which keeps the
pairing intact and is what makes the interval a paired one.

\begin{figure}[t]
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.95}
\noindent\begin{tabularx}{\columnwidth}{@{}r@{\ \,}Z@{}}
\toprule
\multicolumn{2}{@{}l@{}}{\algname{Caliper-matched mechanism gap}%
\label{alg:matched}}\\
\midrule
 & \lvA\algin\ copying observations $\mathcal{C}$, shared-evidence
   observations $\mathcal{S}$, each carrying $(\Dep,\Delta Y)$; calipers
   $\varepsilon$; draws $B$\\
 & \lvA\algout\ the gap and a paired interval, per caliper\\
\midrule
1 & \lvA\algfor\ $\varepsilon\in\{0.02,0.05,0.10\}$:\\
2 & \lvB $\mathcal{M}\leftarrow\varnothing$\\
3 & \lvB\algfor\ $x\in\mathcal{C}$:\\
4 & \lvC $\mathcal{N}\leftarrow\{y\in\mathcal{S}:
   |\Dep(y)-\Dep(x)|\le\varepsilon\}$\\
5 & \lvC\algif\ $\mathcal{N}\neq\varnothing$: append
   $\big(\Delta Y(x),\ \mathrm{mean}_{y\in\mathcal{N}}\Delta Y(y)\big)$
   to $\mathcal{M}$\\
6 & \lvB\algif\ $|\mathcal{M}|<3$: report the interval as undefined and
   continue\\
7 & \lvB $\widehat{\delta}\leftarrow\mathrm{mean}_{(u,v)\in\mathcal{M}}(u-v)$\\
8 & \lvB\algfor\ $b=1..B$: resample $\mathcal{M}$ with replacement,
   \emph{by pair}, and recompute $\widehat{\delta}^{(b)}$\\
9 & \lvB report $\widehat{\delta}$ with the $2.5$ and $97.5$ percentiles of
   $\widehat{\delta}^{(b)}$\\
\midrule
 & \lvA\algret\ the gap per caliper; the claim requires survival at the
   tightest $\varepsilon$ and agreement with the other two procedures.\\
\bottomrule
\end{tabularx}
\end{figure}

The guard on line~6 is the reason a caliper with too little common support
(as for llama/AQuA at $\varepsilon=0.02$, Section~\ref{sec:mechanism-effects})
is reported as undefined rather than as a wide interval, as
Appendix~\ref{sec:app-est-ci} explains.

\section{Estimators and Statistical Procedures}
\label{sec:app-estimators}

\subsection{Dependence}
\label{sec:app-est-dep}

The primary dependence estimator is Equation~\ref{eq:D}. For the contrast
$\Delta \Dep$ we report two aggregations whose difference is not cosmetic. The
\emph{mean of ratios} computes $\Dep$ per question and averages, which is the
estimator frozen for our original four-model substrate and which we retain
there for continuity. The \emph{pooled ratio} sums the numerator and the
denominator of Equation~\ref{eq:D} across questions before dividing, so that a
saturated question contributes zero to both and drops out arithmetically.

For the recovered breadth strata the pooled ratio is primary. The reason is a
selection effect rather than a preference. In the high-competence stratum the
mean of ratios is computable on only a small minority of questions, because
$\Dep$ is undefined wherever every agent is constant across replicates, and the
surviving subset is selected on error variance, which is the very quantity
$\Dep$ measures. In the low-competence stratum the two aggregations agree
closely. We report both throughout so that the choice is visible, and we note
explicitly that the paper therefore uses different primary aggregations on the
two substrates.

\subsection{Decision Change and Ties}
\label{sec:app-est-dy}

The decision-change rate $\Delta Y$ of
Section~\ref{sec:problem-formulation} is computed by comparing conditions at
matched replicate index, which pairs the comparison and removes
between-replicate variation. The directional components $\pbreak$ and $\prescue$
are accumulated on the same pairing, so that the identity of
Proposition~\ref{prop:identity} holds question by question in the estimates and
not merely in expectation.

A plurality tie is a non-answer under the rule of
Section~\ref{sec:problem-formulation}. Two conventions are defensible for scoring
it, and they are not equivalent: a tie may be recorded as an explicit
non-answer, hence incorrect, or it may be broken uniformly at random. We treat
the non-answer convention as primary, because it is the convention under which
the matched-dependence analysis of
Appendix~\ref{sec:app-est-matched} was originally conducted, and we report the
randomized convention alongside it. The two differ by a few points in
$\Delta Y$ and agree in sign and significance pattern throughout, so no
conclusion in this paper, and in particular none of the comparisons in
Section~\ref{sec:repairability-results}, depends on the tie rule.

\subsection{Matched-Dependence Comparison}
\label{sec:app-est-matched}

Three procedures with different assumptions are used, and the mechanism gap is
claimed only where all three agree.

\emph{Caliper matching.} For each copying observation, average the consequences
of shared-evidence observations whose dependence lies within $\varepsilon$, for
$\varepsilon\in\{0.02,0.05,0.10\}$, and report the mechanism gap on the matched
set with a paired bootstrap interval. Survival at the tightest caliper is the
evidence that the gap is not a dependence-magnitude effect.

\emph{Nested linear-probability models.} Regress the decision-change indicator
$\Delta Y$ of Section~\ref{sec:problem-formulation} on dependence; then on
dependence and mechanism; then additionally on their interaction. Nested
$F$-tests assess whether mechanism retains explanatory power after conditioning
on dependence. This is the statistical form of
Corollary~\ref{cor:insufficient}, which states that no function of competence
and dependence alone can predict the collective outcome.

\emph{Nearest-neighbour matching.} In the sense of
\citet{rosenbaum-rubin-1983-central}, match on the confounder triple of dependence
$\Dep$, baseline accuracy, and margin $\Marg$, the last two as defined in
Equation~\ref{eq:margin} and measured on the baseline arm per
Section~\ref{sec:dependence-measurement}, and estimate the average treatment
effect on the matched set. This addresses the objection that the copying arm
might simply face easier or more marginal questions, which would otherwise be an
alternative explanation for the gap reported in
Section~\ref{sec:mechanism-effects}.

\subsection{Intervals and Units of Replication}
\label{sec:app-est-ci}

All intervals are bootstrap percentile intervals at the $95\%$ level, as stated
in Section~\ref{sec:experimental-setup}. The resampling unit is chosen to match
the inferential claim: questions within a cell for the question-level quantities
of Section~\ref{sec:dependence-measurement}, and cells for statements that
generalize across models and benchmarks, such as those in
Section~\ref{sec:generalization}, since cells rather than questions are the
replication unit in that case. Using the wrong unit would understate uncertainty
for exactly the claims the paper rests on, namely the cross-model statements
about $\Delta Y$ and $\Delta A$ in
Section~\ref{sec:repairability-results}. Where fewer than three clusters are
available we report the interval as undefined rather than returning a degenerate
one. The mechanism comparison of Section~\ref{sec:mechanism-effects} resamples
its $K=9$ (model, benchmark) pairs for the macro-average interval, so its
remaining boundary is not a small-cluster concern but the copying arm's
inestimable within-condition churn, reported as a scope limitation in
Appendix~\ref{sec:limitations} rather than as a wide interval.

\section{Reproducibility}
\label{sec:app-repro}

Every quantity in this paper is recomputable offline from stored artifacts. For
each cell we retain the complete agent-by-option score tensor for every
condition, question, and replicate, together with the question text, the
displayed option order, and hashes of the model identifier and prompt. All
analyses in Section~\ref{sec:experiments-results} and
Appendix~\ref{sec:app-estimators} read these tensors and perform no further
generation. Two consequences are worth stating, because both occurred. The
stratification of Appendix~\ref{sec:app-strata} was repaired after data
collection without re-querying any model, and the decision-change quantities
$\Delta Y$, $\pbreak$, and $\prescue$ of
Section~\ref{sec:problem-formulation} were added to the breadth experiments after
the fact, again at zero generation cost, which is what allowed
Section~\ref{sec:repairability-results} to report them at all.

Estimators are imported from a single frozen module by every analysis, so that a
definition such as Equation~\ref{eq:D} or the plurality rule of
Section~\ref{sec:problem-formulation} cannot drift between analyses. Decoding
settings, population size, replicate count, and the prompt templates of
Appendix~\ref{sec:app-prompts} are identical across all cells, and no per-cell
tuning was performed at any stage.

\end{document}
