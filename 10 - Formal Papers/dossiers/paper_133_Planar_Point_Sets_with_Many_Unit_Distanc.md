# Academic Paper Dossier #133: Planar Point Sets with Many Unit Distances

- **Filename**: `unit-distance-proof.pdf`
- **ArXiv Identifier**: `N/A`
- **Document Length**: 18 Pages (46,941 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\unit-distance-proof.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/unit-distance-proof.pdf)

---

## 1. Abstract & Executive Summary
> Abstract For a nite planar set P , let (P ) be the number of unordered unit-distance pairs in P , and let (n) be the maximum of (P ) over all n-point planar sets. We prove that, for some xed >0, one has (n) n1+ for in nitely many n. This disproves the well-known unit distance conjecture from [ Erd46]. The construction passes through an in nite unrami ed tower of totally real number elds with 3-power Galois groups of growing degree, in which a xed set of rational primes splits com- pletely. After adjoining i, these elds produce high-dimensional lattices with many elements whose images under every complex embedding have absolute value 1. Golod Shafarevich theory ensures existence of an in nite such tower, even after a quotient step making the pre-

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
, in an unrami ed pro- 3 setting; see Remark 3.1. Section 2 proves the geometric criterion: given suitable number elds and split primes, it con- structs the planar point sets by embedding norm-one elements into a high-dimensional Minkowski lattice, cutting by a product of discs, and projecting to one complex coordinate. Section 3 then constructs the required elds using the unrami ed pro- 3 tower. Appendix A collects the number- eld conventions and citations used below. Statement on AI Use This problem was solved in a completely automated fashion. Our internal model was given an AI-written statement of the problem, and its output was sent to an AI grading pipeline, which indicated high con dence that the solution was correct. It was only after this point that in- 2 ternal human researchers and mathematicians began to examine the solution carefully. After preliminary AI-assisted veri cation and rewriting, a draft was sent to external mathematicians, including several number theory experts, who con rmed the proof s correctness (and have already simpli ed and strengthened the argument). The present manuscript is a human-edited exposi- tion of the autonomously produced solution, with references, reorganized proofs, and additional explanatory material added afterward. The original AI-written prompt given to the internal model was: Prompt. Let P R2 be a nite set of distinct points. De ne (P ) = { {p,q } ( P 2 ) : p q 2 = 1 } and, for each integer n 1, (n) = max P R2 |P |=n (P ). Resolve Erd s s planar unit-distance problem completely: (n) n1+O(1/ log log n) asn ? Equivalently, determine whether there exist absolute constants C >0 and N N such that (n) n1+C/ log log n for every integer n N . Here log denotes the natural logarithm, and N may be taken large enough that log logn >0. The hidden constant in O(1/ log logn) is absolute and independent of P and n. A complete solution must prove exactly one of the following. A rmative resolution. Prove that there exist absolute constants C >0 andN N such that every set P R2 of n N distinct points satis es (P ) n1+C/ log log n. Negative resolution. Prove that no such constants exist. Equivalently, prove that for every C >0 and every N N, there are an integer n N and a set P R2 of n distinct points such that (P )>n 1+C/ log log n. Pairs are unordered, the distance is the usual Euclidean distance in R2, and the asymptotic assertion is for all su ciently large integers n, not merely in nitely many n. Partial progress does not

### Key Mathematical Formulations Extracted from Body:
- `$p = 3 in the tower construction (this avoids all dyadic in nite-prime$`
- `$M = L1 L$`
- `$D =O( log )$`
- `$F =F0 F1 F2$`
- `$j = [Fj : Q]$`
- `$j =Fj(i), with complex conjugation c$`

## 4. Empirical Evaluation & Benchmark Evidence
Analyzed across document pages.

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
