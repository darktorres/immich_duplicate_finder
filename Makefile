.PHONY: help install test test-cov lint format check clean run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies using Poetry
	poetry install



test: ## Run tests
	poetry run pytest -v

test-cov: ## Run tests with coverage
	poetry run pytest --cov=. --cov-report=html --cov-report=term-missing -v

test-watch: ## Run tests in watch mode
	poetry run pytest -f -v

lint: ## Run linting
	poetry run ruff check --fix .

format: ## Format code
	poetry run ruff format .

check: ## Check code quality without making changes
	poetry run ruff check . && poetry run ruff format --diff .

clean: ## Clean up generated files
	rm -rf .coverage htmlcov/ .pytest_cache/ __pycache__/ *.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Run the Streamlit application
	poetry run streamlit run app.py



dev: ## Set up development environment
	poetry install
	poetry run pre-commit install

# Docker targets
docker-build: ## Build Docker image
	docker build -t duplicate-finder .

docker-run: ## Run Docker container
	docker run -p 8501:8501 duplicate-finder

# CI targets
ci-test: ## Run CI tests locally
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run pytest --cov=. --cov-report=xml --cov-report=term-missing -v