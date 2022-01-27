.PHONY: help backup clean install run
.DEFAULT_GOAL := help

help:
	@awk 'BEGIN {FS = ":.*#"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z0-9_-]+:.*?#/ { printf "  \033[36m%-27s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST); printf "\n"

backup: ## Backup codebase (*_YYYYMMDD_HHMM.tar.gz)
backup: archive=`pwd`_`date +'%Y%m%d_%H%M'`.tar.gz
backup:
	@tar -czf $(archive) --exclude=__pycache__ *
	@ls -l `pwd`*.tar.gz

clean: ## Clean codebase
	@rm -fR __pycache__
	@rm -fR models/__pycache__

install: ## Install requirements
	pip install -r requirements.txt

run: ## Run the Machine Learning model test
	python main.py
