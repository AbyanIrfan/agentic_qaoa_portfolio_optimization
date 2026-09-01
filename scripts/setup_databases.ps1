# Rebuilds the arXiv corpus from scratch and backfills journal_ref/doi metadata.
# Usage: .\scripts\setup_databases.ps1

python scripts\setup_arxiv_db.py `
  --index_name "qaoa-portfolio-v1" `
  --chunk_size 1024 `
  --max_results_per_request 100 `
  --max_total_results 350 `
  --init_db true `
  --category "quant-ph,q-fin.PM,q-fin.CP" `
  --keyword_groups "QAOA,quantum approximate optimization algorithm,warm-start QAOA,XY mixer,digitized counterdiabatic quantum optimization;portfolio optimization,cardinality-constrained,asset allocation,Markowitz,CVaR" `
  --date_range "20140101-20260819" `
  --seed_ids "2606.13380,1904.07358" `
  --pdf_dir "data/papers/arxiv" `
  --db_dir "data/references/arxiv_db"

python scripts\backfill_journal_refs.py `
  --index_name "qaoa-portfolio-v1" `
  --db_dir "data/references/arxiv_db"

  # Optional: regenerates the Qiskit doc reference store from the installed
# qiskit/qiskit-optimization/qiskit-algorithms/qiskit-finance packages.
# Skip this if data/references/qiskit_raw.json and qiskit_formatted.json
# are already present (e.g. pulled from an existing corpus).
python scripts\setup_qiskit_db.py `
  --save_raw_json_path="data/references/qiskit_raw.json" `
  --save_formatted_json_path="data/references/qiskit_formatted.json"