#!/usr/bin/env bash
# Reverses the 2026-08-30 root tidy-up. Run from the project root:
#   bash claude_generated_files_and_plans/UNDO_move.sh
set -e
cd "$(dirname "$0")/.."
G=claude_generated_files_and_plans

for f in 01_UIL_SYSTEM_MONOGRAPH 02_THEORETICAL_FOUNDATIONS_AND_LITERATURE_SURVEY \
         03_EMPIRICAL_EVALUATION_AND_FAILURE_MODE_ANALYSIS \
         04_HIERARCHICAL_MEMORY_AND_KNOWLEDGE_ROUTED_NETWORK_DESIGN \
         05_NOVELTY_IP_ANALYSIS_AND_FUTURE_RESEARCH_ROADMAP; do
  mv "academic_v2/$f.docx" "./$f.docx"
done

for f in COGNITIVE_ENGINE_RESCUE.md RESEARCH_FAILURE_ANALYSIS.md HARDCODED_VALUE_AUDIT_QA.md \
         LITSURVEY_CORRECTIONS.md DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md; do
  mv "$G/analysis/$f" "./$f"
done

for f in LITERATURE_SURVEY.md LITERATURE_SURVEY_PAPERS.md PROFESSOR_PROJECT_BRIEF.md \
         PROFESSOR_PROJECT_BRIEF_V2.md uil_ip_strategy.md; do
  mv "$G/research_docs/$f" "./$f"
done

mv "$G/unrelated_lca/agent_architecture.md" ./agent_architecture.md
mv "$G/unrelated_lca/lca_uil_analysis.md"   ./lca_uil_analysis.md

mv "$G/documents/Universal Intelligence Layer.docx" "./Universal Intelligence Layer.docx"
mv "$G/documents/UIL System Refinement.docx"        "./UIL System Refinement.docx"
mv "$G/documents/DAPTIVE.docx"                      "./DAPTIVE.docx"

mv "$G/scratch/new.text"        ./new.text
mv "$G/scratch/_wtest.md"       academic_v2/_wtest.md
mv "$G/scratch/_build_doc.py"   academic_v2/_build_doc.py
mv "$G/scratch/_parts_t1.md"    academic_v2/_parts/t1.md

mv results/_leaked_from_tests/uil_formula_changelog_v20.json tests/uil_formula_changelog_v20.json
rmdir results/_leaked_from_tests 2>/dev/null || true

mv _recovery/_recover_site.py ./_recover_site.py
sed -i 's|^ROOT = Path(__file__).resolve().parent  # == _recovery/ (script moved here 2026-08-30)$|ROOT = Path(__file__).resolve().parent|' _recover_site.py
sed -i 's|^OUT = ROOT$|OUT = ROOT / "_recovery"|' _recover_site.py

echo "Restored. $G/plans/ holds COPIES only (originals in ~/.claude/plans/);"
echo "delete $G manually once you have checked nothing else was put in it."

# Restore CLAUDE.md to its pre-tidy state (path rewrites + the relocation note).
cp "$G/CLAUDE.md.pre-tidy-2026-08-30.bak" CLAUDE.md
echo "CLAUDE.md restored from backup."
