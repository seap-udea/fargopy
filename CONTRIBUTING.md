# Contributing to FARGOpy

We welcome contributions to FARGOpy! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fargopy.git
   cd fargopy
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   make test
   ```

3. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub

## Code Standards

- Follow PEP 8 style guidelines
- Write docstrings for all public functions and classes
- Add unit tests for new functionality
- Ensure all tests pass before submitting a PR

## Testing

Run tests using:
```bash
make test
```

Or directly with pytest:
```bash
pytest
```

## Documentation

- Update documentation for any new features
- Ensure docstrings are complete and accurate
- Build docs locally to verify formatting:
  ```bash
  make docs
  ```

## Questions?

Contact the maintainers:
- Jorge I. Zuluaga: jorge.zuluaga@udea.edu.co
- Alejandro Murillo-González: alejandro.murillo1@udea.edu.co
- Matías Montesinos: matias.montesinosa@usm.cl

Thank you for contributing to FARGOpy!
