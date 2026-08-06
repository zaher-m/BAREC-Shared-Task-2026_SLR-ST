PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: help setup verify preprocess infer validate clean-pyc

help:
	@echo "verify      recompute the reported numbers from the tracked caches (start here)"
	@echo "setup       create .venv and install the package (editable)"
	@echo "preprocess  build data/proc_raw_* and data/proc_d3tok_* from the BAREC parquets"
	@echo "infer       run the shipped 21-member ensemble on the blind set"
	@echo "validate    check a submission file against the CodaBench format"
	@echo "clean-pyc   remove __pycache__ trees"
	@echo
	@echo "Training a member (see experiments/ for the launchers that built each family):"
	@echo "  \$$(PY) -m slra_st.train --model aubmindlab/bert-base-arabertv2 \\"
	@echo "      --tag arabertv2_emd_d3 --variant d3tok --objective emd --bs 32 --lr 2e-5"

setup:
	python -m venv .venv
	.venv/bin/pip install -e .

ARGS ?=

verify:
	$(PY) -m slra_st.verify $(ARGS)

preprocess:
	$(PY) -m slra_st.preprocess --variant raw
	$(PY) -m slra_st.preprocess --variant d3tok

ENSEMBLE ?= final_v8
OUT ?= artifacts/predictions/blind_prediction

infer:
	$(PY) -m slra_st.infer --input data/blind_sent.parquet --ensemble $(ENSEMBLE) --out $(OUT)

FILE ?= submissions/submitted/pred_priorQWK_acc

validate:
	$(PY) -m slra_st.submission $(FILE)

clean-pyc:
	find src experiments -name __pycache__ -type d -prune -exec rm -rf {} +
