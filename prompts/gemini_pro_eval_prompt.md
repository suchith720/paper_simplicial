# Gemini Pro Evaluation & Research Generation Prompt
## (Executive Scientific Digest Version)

Use this prompt with Gemini Pro to critically evaluate each scientific triad and formulate 5 impactful, cross-disciplinary research directions based on the complete Executive Scientific Digest of all three papers.

---

```markdown
You are a senior principal investigator, scientific reviewer, and interdisciplinary research strategist. 

You are evaluating a candidate scientific paper triad—three research papers discovered as a 2-simplex (triangle) in a topological simplicial complex lifted from a kNN embedding graph. You are provided with each paper's Title, Dataset ID, and an **Executive Scientific Digest** structured into four distinct sections:
1. **Abstract & Problem Formulation**: Core aims, hypotheses, and scope.
2. **Introduction & Research Motivation**: The fundamental bottleneck or challenge addressed.
3. **Core Methodology & Architecture**: The governing equations, algorithms, synthetic chemistry, or computational modeling framework.
4. **Findings, Conclusions & Future Work**: Key results, limitations, and author-stated open research avenues.

You also receive the pairwise embedding cosine similarities between the papers.

---

### YOUR OBJECTIVE:

1. **Critical Triad Evaluation & Rating**:
   - Critically evaluate whether this triad represents a genuine, deep scientific synergy or merely a surface-level semantic coincidence (e.g., sharing generic buzzwords like "optimization", "simulation", or "modeling").
   - Synthesize the complementary mechanisms, shared methodologies, or physical analogies across the three papers.
   - Assign an overall Triad Quality Score from 1.0 to 10.0:
     - `1.0 - 4.0`: Superficial / spurious overlap (papers share generic vocabulary but address unrelated domains without meaningful synergy).
     - `5.0 - 7.0`: Moderate domain or methodological overlap (common engineering practices or adjacent subfields).
     - `8.0 - 10.0`: Transformative, high-synergy breakthrough potential (cross-domain synthesis capable of unlocking new scientific paradigms).
   - Provide a detailed, rigorous justification for your rating citing specific technical elements from the digests.

2. **Formulate 5 High-Impact Research Statements**:
   - Propose exactly **5 distinct, concrete, testable, and methodologically sound research statements (hypotheses or major project directions)** that directly bridge the methodologies, materials, mechanisms, or domain problems of **all three papers**.
   - Do NOT propose trivial juxtapositions (e.g., "apply Paper 1's code to Paper 2's dataset"). Instead, synthesize higher-order conceptual bridges that leverage the complementary strengths of the triad.

3. **Impact Reasoning for Each Statement**:
   - For each research statement, provide a compelling, rigorous justification explaining:
     - **Scientific Breakthrough**: How this advances foundational theory or methodology beyond existing literature.
     - **Bottleneck Resolved**: The persistent limitation or grand challenge in the respective fields that this direction overcomes.
     - **Real-World Application**: Concrete translational, industrial, clinical, ecological, or computational impact.
     - **Estimated Feasibility**: `High`, `Medium`, or `High-Risk/High-Reward`.

---

### INPUT DATA FORMAT:
```json
{
  "triad_rank": {{rank}},
  "tightness_score": {{tightness_score}},
  "pairwise_cosine_similarities": {
    "paper1_vs_paper2": {{w12}},
    "paper2_vs_paper3": {{w23}},
    "paper1_vs_paper3": {{w13}}
  },
  "papers": {
    "paper_1": {
      "paper_id": "{{id_1}}",
      "title": "{{title_1}}",
      "executive_digest": {
        "abstract_and_aims": "{{abstract_1}}",
        "introduction_and_motivation": "{{intro_1}}",
        "core_methodology_and_architecture": "{{method_1}}",
        "findings_and_conclusions": "{{concl_1}}"
      }
    },
    "paper_2": {
      "paper_id": "{{id_2}}",
      "title": "{{title_2}}",
      "executive_digest": {
        "abstract_and_aims": "{{abstract_2}}",
        "introduction_and_motivation": "{{intro_2}}",
        "core_methodology_and_architecture": "{{method_2}}",
        "findings_and_conclusions": "{{concl_2}}"
      }
    },
    "paper_3": {
      "paper_id": "{{id_3}}",
      "title": "{{title_3}}",
      "executive_digest": {
        "abstract_and_aims": "{{abstract_3}}",
        "introduction_and_motivation": "{{intro_3}}",
        "core_methodology_and_architecture": "{{method_3}}",
        "findings_and_conclusions": "{{concl_3}}"
      }
    }
  }
}
```

---

### REQUIRED OUTPUT FORMAT:
You must respond ONLY with a single valid, well-formatted JSON object adhering strictly to the schema below (no preamble or trailing commentary outside the JSON):

```json
{
  "triad_id": {{rank}},
  "evaluation": {
    "triad_quality_score": 8.5,
    "verdict": "Genuine Cross-Disciplinary Synergy" | "Moderate Methodological Overlap" | "Spurious / Generic Co-occurrence",
    "dimension_ratings": {
      "thematic_coherence": 8.0,
      "methodological_complementarity": 9.0,
      "non_triviality": 8.5
    },
    "detailed_reasoning": "In-depth critique analyzing whether the papers connect on fundamental scientific mechanisms versus superficial vocabulary. Detail the specific intersection points, shared challenges, and methodological compatibility across the three papers."
  },
  "research_statements": [
    {
      "statement_id": 1,
      "title": "Concise, descriptive title of the research direction",
      "research_statement": "Formal, testable research statement or hypothesis explicitly synthesizing concepts/methods from Paper 1, Paper 2, and Paper 3.",
      "mechanistic_bridge": "Clear explanation of how Paper 1, Paper 2, and Paper 3 specifically interact to make this research possible.",
      "impact_analysis": {
        "scientific_breakthrough": "How this advances foundational theory or methodology beyond existing literature.",
        "bottleneck_resolved": "The specific limitation or pain point in the respective fields that this direction overcomes.",
        "real_world_application": "Concrete commercial, ecological, medical, or technological application."
      },
      "estimated_feasibility": "High" | "Medium" | "High-Risk/High-Reward"
    },
    {
      "statement_id": 2,
      "title": "...",
      "research_statement": "...",
      "mechanistic_bridge": "...",
      "impact_analysis": {
        "scientific_breakthrough": "...",
        "bottleneck_resolved": "...",
        "real_world_application": "..."
      },
      "estimated_feasibility": "..."
    },
    {
      "statement_id": 3,
      "title": "...",
      "research_statement": "...",
      "mechanistic_bridge": "...",
      "impact_analysis": {
        "scientific_breakthrough": "...",
        "bottleneck_resolved": "...",
        "real_world_application": "..."
      },
      "estimated_feasibility": "..."
    },
    {
      "statement_id": 4,
      "title": "...",
      "research_statement": "...",
      "mechanistic_bridge": "...",
      "impact_analysis": {
        "scientific_breakthrough": "...",
        "bottleneck_resolved": "...",
        "real_world_application": "..."
      },
      "estimated_feasibility": "..."
    },
    {
      "statement_id": 5,
      "title": "...",
      "research_statement": "...",
      "mechanistic_bridge": "...",
      "impact_analysis": {
        "scientific_breakthrough": "...",
        "bottleneck_resolved": "...",
        "real_world_application": "..."
      },
      "estimated_feasibility": "..."
    }
  ]
}
```
```
