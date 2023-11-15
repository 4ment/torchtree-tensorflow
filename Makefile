install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall torchtree-tensorflow

lint: FORCE
	flake8 --exit-zero torchflow test
	isort --check .
	black --check .

format: FORCE
	isort .
	black .

test: FORCE
	pytest

clean: FORCE
	rm -fr torchflow/__pycache__ test/__pycache__

nuke: FORCE
	git clean -dfx -e torchtree_tensorflow.egg-info

	done

FORCE: