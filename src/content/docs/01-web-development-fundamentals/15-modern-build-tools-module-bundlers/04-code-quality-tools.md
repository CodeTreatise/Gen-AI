---
title: "Code Quality Tools"
---

# Code Quality Tools

## Introduction

Automated code quality tools catch bugs, enforce consistency, and save code review time. Linters find errors, formatters ensure consistent style, and git hooks prevent bad commits.

This lesson covers ESLint, Prettier, Husky, lint-staged, and EditorConfig.

### What We'll Cover

- ESLint for linting
- Prettier for formatting
- ESLint + Prettier integration
- Husky for git hooks
- lint-staged for efficiency
- EditorConfig for consistency

### Prerequisites

- npm/package management
- Basic git knowledge
- JavaScript/TypeScript basics

---

## ESLint

### What ESLint Does

ESLint catches errors and enforces code patterns:

```javascript
// ESLint catches these issues:

const x = 5;     // ❌ 'x' is never used
x = 6            // ❌ Missing semicolon (if enabled)
console.log(y);  // ❌ 'y' is not defined
```

### Setup

```bash
npm init @eslint/config
# or
npm install -D eslint
npx eslint --init
```

### Configuration

```javascript
// eslint.config.js (Flat Config - ESLint 9+)
import js from '@eslint/js';

export default [
  js.configs.recommended,
  {
    files: ['**/*.js'],
    rules: {
      'no-unused-vars': 'warn',
      'no-console': 'warn',
      'eqeqeq': 'error',
      'curly': 'error'
    }
  },
  {
    ignores: ['dist/**', 'node_modules/**']
  }
];
```

### Legacy Config (.eslintrc)

```json
{
  "env": {
    "browser": true,
    "es2024": true,
    "node": true
  },
  "extends": ["eslint:recommended"],
  "rules": {
    "no-unused-vars": "warn",
    "no-console": "warn",
    "eqeqeq": "error"
  }
}
```

### Rule Levels

| Level | Meaning |
|-------|---------|
| `"off"` or `0` | Disable rule |
| `"warn"` or `1` | Warning (doesn't fail) |
| `"error"` or `2` | Error (fails lint) |

### Running ESLint

```bash
# Lint files
npx eslint src/

# Lint and fix
npx eslint src/ --fix

# Specific files
npx eslint "src/**/*.{js,ts}"
```

### TypeScript ESLint

```bash
npm install -D typescript-eslint
```

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      '@typescript-eslint/no-unused-vars': 'warn'
    }
  }
);
```

---

## Prettier

### What Prettier Does

Prettier formats code automatically—no decisions needed:

```javascript
// Before Prettier
const user={name:"John",age:30,email:"john@example.com"}

// After Prettier
const user = {
  name: "John",
  age: 30,
  email: "john@example.com",
};
```

### Setup

```bash
npm install -D prettier
```

### Configuration

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 80,
  "bracketSpacing": true,
  "arrowParens": "always"
}
```

### Ignore Files

```text
# .prettierignore
dist
node_modules
*.min.js
package-lock.json
```

### Running Prettier

```bash
# Check formatting
npx prettier --check .

# Format files
npx prettier --write .

# Format specific files
npx prettier --write "src/**/*.{js,ts,jsx,tsx}"
```

---

## ESLint + Prettier Integration

### The Problem

ESLint and Prettier can conflict on formatting rules.

### The Solution

```bash
npm install -D eslint-config-prettier
```

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import prettier from 'eslint-config-prettier';

export default [
  eslint.configs.recommended,
  prettier  // Must be last - disables conflicting rules
];
```

### Recommended Setup

| Tool | Responsibility |
|------|----------------|
| **ESLint** | Code quality, bugs, patterns |
| **Prettier** | All formatting |

```json
// package.json scripts
{
  "scripts": {
    "lint": "eslint .",
    "lint:fix": "eslint . --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check ."
  }
}
```

---

## Husky

### What Husky Does

Husky runs scripts on git hooks (pre-commit, pre-push, etc.):

```
git commit
    │
    ▼
pre-commit hook (Husky)
    │
    ├── Run linter
    ├── Run formatter
    └── Run tests
    │
    ▼
Commit succeeds (if all pass)
```

### Setup

```bash
npm install -D husky
npx husky init
```

This creates `.husky/` directory and adds `prepare` script.

### Add Pre-commit Hook

```bash
# .husky/pre-commit
npm run lint
npm run format:check
```

### Add Pre-push Hook

```bash
# Create hook
echo "npm test" > .husky/pre-push
chmod +x .husky/pre-push
```

---

## lint-staged

### The Problem

Running linter on entire project is slow:

```bash
# Slow: lints everything
npx eslint .
```

### The Solution

lint-staged runs tools only on staged files:

```bash
# Fast: only lints files you're committing
npx lint-staged
```

### Setup

```bash
npm install -D lint-staged
```

```json
// package.json
{
  "lint-staged": {
    "*.{js,ts,jsx,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss}": [
      "prettier --write"
    ],
    "*.{json,md}": [
      "prettier --write"
    ]
  }
}
```

### Integrate with Husky

```bash
# .husky/pre-commit
npx lint-staged
```

### Full Setup

```json
// package.json
{
  "scripts": {
    "prepare": "husky",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "lint-staged": {
    "*.{js,ts,jsx,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,json,md}": ["prettier --write"]
  }
}
```

---

## EditorConfig

### What EditorConfig Does

EditorConfig ensures consistent settings across editors (VS Code, Vim, etc.):

```ini
# .editorconfig
root = true

[*]
indent_style = space
indent_size = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

### How It Works

1. Install EditorConfig extension in your editor
2. Create `.editorconfig` in project root
3. Editor automatically uses these settings

---

## Complete Project Setup

### Directory Structure

```
project/
├── .editorconfig
├── .eslintignore
├── .gitignore
├── .husky/
│   └── pre-commit
├── .prettierignore
├── .prettierrc
├── eslint.config.js
├── package.json
└── src/
```

### package.json

```json
{
  "name": "my-project",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "lint:fix": "eslint . --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "typecheck": "tsc --noEmit",
    "validate": "npm run lint && npm run format:check && npm run typecheck",
    "prepare": "husky"
  },
  "devDependencies": {
    "@eslint/js": "^9.0.0",
    "eslint": "^9.0.0",
    "eslint-config-prettier": "^9.0.0",
    "husky": "^9.0.0",
    "lint-staged": "^15.0.0",
    "prettier": "^3.0.0",
    "typescript": "^5.0.0",
    "typescript-eslint": "^7.0.0",
    "vite": "^5.0.0"
  },
  "lint-staged": {
    "*.{js,ts,jsx,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,json,md}": ["prettier --write"]
  }
}
```

### Quick Setup Commands

```bash
# Initialize project
npm init -y

# Install all quality tools
npm install -D eslint prettier husky lint-staged eslint-config-prettier

# Initialize ESLint
npm init @eslint/config

# Initialize Husky
npx husky init

# Create pre-commit hook
echo "npx lint-staged" > .husky/pre-commit
```

---

## VS Code Integration

### Extensions to Install

- ESLint
- Prettier - Code formatter
- EditorConfig for VS Code

### Settings

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

---

## Hands-on Exercise

### Your Task

Set up a complete code quality pipeline:

1. Create a new project
2. Configure ESLint and Prettier
3. Set up Husky with lint-staged
4. Add a pre-commit hook
5. Test with a commit

<details>
<summary>✅ Solution</summary>

```bash
# Create project
mkdir quality-demo && cd quality-demo
npm init -y

# Install tools
npm install -D eslint prettier husky lint-staged eslint-config-prettier @eslint/js

# Create ESLint config
cat > eslint.config.js << 'EOF'
import eslint from '@eslint/js';
import prettier from 'eslint-config-prettier';

export default [
  eslint.configs.recommended,
  prettier,
  {
    rules: {
      'no-unused-vars': 'warn',
      'no-console': 'warn'
    }
  },
  {
    ignores: ['dist/**', 'node_modules/**']
  }
];
EOF

# Create Prettier config
cat > .prettierrc << 'EOF'
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5"
}
EOF

# Initialize Husky
npx husky init

# Create pre-commit hook
echo "npx lint-staged" > .husky/pre-commit
chmod +x .husky/pre-commit

# Add lint-staged to package.json
npm pkg set lint-staged='{"*.{js,ts}": ["eslint --fix", "prettier --write"]}'

# Add scripts
npm pkg set scripts.lint="eslint ."
npm pkg set scripts.format="prettier --write ."

# Create test file
mkdir src
cat > src/index.js << 'EOF'
const message = "Hello World"
console.log(message);
EOF

# Initialize git
git init
git add .

# Test the hook
git commit -m "Initial commit"
```
</details>

---

## Summary

✅ **ESLint** catches bugs and enforces code patterns
✅ **Prettier** handles all formatting—no debates
✅ Use **eslint-config-prettier** to avoid conflicts
✅ **Husky** runs scripts on git hooks
✅ **lint-staged** runs tools only on staged files for speed
✅ **EditorConfig** ensures consistent editor settings
✅ Configure **VS Code** for format-on-save

**Back to:** [Build Tools Overview](./00-modern-build-tools.md)

---

## Further Reading

- [ESLint Documentation](https://eslint.org/)
- [Prettier Documentation](https://prettier.io/)
- [Husky Documentation](https://typicode.github.io/husky/)
- [lint-staged Documentation](https://github.com/lint-staged/lint-staged)

<!-- 
Sources Consulted:
- ESLint docs: https://eslint.org/
- Prettier docs: https://prettier.io/
-->
