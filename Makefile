# Virtual environment name
VENV_NAME = .venv

# Requirements file
REQUIREMENTS_FILE = requirements.txt
REQUIREMENTS_DEV_FILE = requirements.dev.txt

# Source code directory
SRC_DIR = reader_vl

# Formatting tool
FORMATTING_TOOL = black

# Static type checker
TYPE_CHECKER = mypy

# Create virtual environment
.venv:
	python3 -m venv $(VENV_NAME)

# Install dependencies
install:
	pip install -r $(REQUIREMENTS_FILE)

# Install dev dependencies (linting, formatting, testing)
install-dev:
	pip install --upgrade -r $(REQUIREMENTS_DEV_FILE)

# Run linters
lint:
	pylint $(SRC_DIR) 
	flake8 $(SRC_DIR)

# Run formatter
format:
	$(FORMATTING_TOOL) $(SRC_DIR) $(TEST_DIR)

# Run type checker
type-check:
	$(TYPE_CHECKER) $(SRC_DIR)

# Clean up virtual environment
clean:
	rm -rf $(VENV_NAME)

# Pre-commit hook (example)
pre-commit: 
	pre-commit run --all-files