---
title: "Version Control with Git"
---

# Version Control with Git

## Introduction

Every professional developer uses version control. **Git** tracks changes to your code, enables collaboration, and provides a safety net when things go wrong. Combined with platforms like GitHub, GitLab, or Bitbucket, Git enables powerful workflows for teams of any size.

This lesson covers Git fundamentals: tracking changes, branching, collaborating through pull requests, and best practices for commit hygiene.

### What We'll Cover

- Git fundamentals (init, add, commit)
- Branching and merging
- Remote repositories
- Pull requests workflow
- .gitignore for secrets and dependencies
- Commit message conventions
- Basic conflict resolution

### Prerequisites

- Command line basics
- Git installed (`git --version` to verify)

---

## Git Fundamentals

### What Git Does

Git tracks **snapshots** of your project over time:

```
Project History:
                                           
  Initial ──► Add Login ──► Fix Bug ──► Add Dashboard ──► NOW
  (commit 1)   (commit 2)   (commit 3)    (commit 4)
                                           
  You can go back to any point in time!
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Repository** | Project folder with Git history |
| **Commit** | Snapshot of changes with message |
| **Branch** | Parallel line of development |
| **Remote** | Server copy of repository (GitHub) |
| **Working Directory** | Files you see and edit |
| **Staging Area** | Changes ready to commit |

### Initializing a Repository

```bash
# Create new repository
cd my-project
git init

# Or clone an existing one
git clone https://github.com/username/repo.git
```

### The Three Areas

```
┌─────────────────────────────────────────────────────────┐
│                    Working Directory                    │
│                (files you see and edit)                 │
└─────────────────────────┬───────────────────────────────┘
                          │ git add
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     Staging Area                        │
│                 (changes ready to commit)               │
└─────────────────────────┬───────────────────────────────┘
                          │ git commit
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      Repository                         │
│                  (committed history)                    │
└─────────────────────────────────────────────────────────┘
```

### Basic Workflow

```bash
# 1. Check status
git status

# 2. Stage changes
git add filename.js       # Stage specific file
git add .                 # Stage all changes

# 3. Commit with message
git commit -m "Add user login feature"

# 4. View history
git log --oneline
```

---

## Staging and Committing

### git add - Stage Changes

```bash
# Stage specific files
git add index.html
git add src/app.js src/utils.js

# Stage all changes in current directory
git add .

# Stage all changes in entire repo
git add -A

# Interactive staging
git add -p  # Review and stage hunks
```

### git commit - Save Snapshot

```bash
# Commit with inline message
git commit -m "Fix navigation bug"

# Commit with editor (for longer messages)
git commit

# Amend last commit (before pushing)
git commit --amend -m "Better message"

# Add to last commit without changing message
git commit --amend --no-edit
```

### git status - Check State

```bash
$ git status
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   app.js        ← Staged

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
        modified:   styles.css    ← Modified but not staged

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        newfile.js                ← New, not tracked
```

### git diff - See Changes

```bash
# Changes not yet staged
git diff

# Changes staged but not committed
git diff --staged

# Changes between commits
git diff abc123 def456
```

---

## Branching and Merging

Branches let you work on features without affecting the main code.

### Why Branch?

```
main:     ──A──B──C──────────────────F──
                  ╲                  ╱
feature:           ──D──E──────────
                          ╲
bugfix:                    ──G──H──
```

### Branch Commands

```bash
# List branches
git branch           # Local branches
git branch -a        # All branches (including remote)

# Create branch
git branch feature-login

# Switch to branch
git checkout feature-login
# Or (newer syntax)
git switch feature-login

# Create and switch in one command
git checkout -b feature-login
# Or
git switch -c feature-login

# Delete branch
git branch -d feature-login  # Safe delete (must be merged)
git branch -D feature-login  # Force delete
```

### Merging

Combine branches:

```bash
# Switch to target branch (usually main)
git checkout main

# Merge feature branch
git merge feature-login
```

### Merge Types

**Fast-forward merge** (no new commit):
```
Before:  main: A──B
         feature:  ──C──D

After:   main: A──B──C──D
```

**Three-way merge** (creates merge commit):
```
Before:  main: A──B──E
         feature:  ╲──C──D

After:   main: A──B──E──────M
               ╲         ╱
         feature: ──C──D──
```

---

## Remote Repositories

Share code and collaborate.

### Adding a Remote

```bash
# Add remote (usually named 'origin')
git remote add origin https://github.com/username/repo.git

# View remotes
git remote -v
```

### Push and Pull

```bash
# Push to remote
git push origin main

# Push and set upstream (first time)
git push -u origin main

# After setting upstream, just:
git push

# Pull changes from remote
git pull origin main
# Or just:
git pull
```

### Fetch vs Pull

```bash
# Fetch: Download but don't merge
git fetch origin

# Pull: Fetch + Merge
git pull origin main
# Equivalent to:
git fetch origin
git merge origin/main
```

---

## Pull Requests Workflow

The standard collaboration pattern for teams.

### The Workflow

```
1. Create branch      git checkout -b feature-x
2. Make changes       (edit files)
3. Commit             git commit -m "Add feature X"
4. Push               git push -u origin feature-x
5. Open PR            (on GitHub/GitLab)
6. Review             (team reviews code)
7. Merge              (click Merge button)
8. Clean up           git branch -d feature-x
```

### Best Practices

| Practice | Why |
|----------|-----|
| Small PRs | Easier to review |
| Descriptive titles | Clear purpose |
| Link to issues | Traceability |
| Self-review first | Catch obvious issues |
| Respond to feedback | Collaborate effectively |

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Tests added/updated
- [ ] Documentation updated
```

---

## .gitignore

Prevent certain files from being tracked.

### Common .gitignore

```gitignore
# Dependencies
node_modules/
vendor/
__pycache__/

# Build outputs
dist/
build/
*.bundle.js

# Environment and secrets
.env
.env.local
*.pem
secrets.json

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Test coverage
coverage/
```

### Patterns

| Pattern | Matches |
|---------|---------|
| `*.log` | All .log files |
| `logs/` | logs directory |
| `/debug.log` | debug.log in root only |
| `**/temp` | temp folder anywhere |
| `!important.log` | Exception - track this file |

### Already Tracked Files

If a file is already tracked, adding to .gitignore won't remove it:

```bash
# Remove from tracking (but keep file)
git rm --cached filename

# Remove from tracking (directory)
git rm -r --cached node_modules/

# Then commit
git commit -m "Remove node_modules from tracking"
```

---

## Commit Message Conventions

Good commit messages are essential for project history.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Conventional Commits

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (not CSS) |
| `refactor` | Code restructure |
| `test` | Adding tests |
| `chore` | Build, deps, config |

### Examples

```bash
# Good
git commit -m "feat(auth): add password reset flow"
git commit -m "fix(api): handle timeout errors in fetchUser"
git commit -m "docs: update API documentation"

# Bad
git commit -m "fixed stuff"
git commit -m "wip"
git commit -m "asdfasdf"
```

### Guidelines

- Use imperative mood: "Add feature" not "Added feature"
- Limit subject to 50 characters
- No period at end of subject
- Body explains what and why, not how

---

## Conflict Resolution

When two branches modify the same lines, Git can't auto-merge.

### When Conflicts Happen

```bash
$ git merge feature-branch
Auto-merging app.js
CONFLICT (content): Merge conflict in app.js
Automatic merge failed; fix conflicts and then commit.
```

### Conflict Markers

```javascript
function getData() {
<<<<<<< HEAD
  return fetch('/api/v2/data');
=======
  return fetch('/api/data').then(r => r.json());
>>>>>>> feature-branch
}
```

- `<<<<<<< HEAD` = Your current branch code
- `=======` = Divider
- `>>>>>>> feature-branch` = Incoming branch code

### Resolving Conflicts

1. Open the conflicted file
2. Choose which code to keep (or combine both)
3. Remove the conflict markers
4. Stage and commit

```javascript
// After resolution:
function getData() {
  return fetch('/api/v2/data').then(r => r.json());
}
```

```bash
git add app.js
git commit -m "Resolve merge conflict in app.js"
```

### VS Code Conflict Resolution

VS Code provides helpful buttons:
- **Accept Current Change** - Keep HEAD
- **Accept Incoming Change** - Keep feature branch
- **Accept Both Changes** - Keep both
- **Compare Changes** - Side-by-side view

---

## Common Git Commands Reference

### Daily Use

```bash
git status              # Check status
git add .               # Stage all
git commit -m "msg"     # Commit
git push                # Push to remote
git pull                # Pull from remote
git log --oneline       # View history
```

### Branching

```bash
git branch              # List branches
git checkout -b name    # Create & switch
git checkout main       # Switch to main
git merge name          # Merge branch
git branch -d name      # Delete branch
```

### Undo Changes

```bash
git restore file        # Discard changes (not staged)
git restore --staged file  # Unstage
git reset --soft HEAD~1    # Undo commit, keep changes staged
git reset --hard HEAD~1    # Undo commit, discard changes
git revert abc123          # Create commit that undoes abc123
```

### Stash

```bash
git stash               # Save changes temporarily
git stash pop           # Restore and remove stash
git stash list          # List stashes
git stash drop          # Remove stash
```

---

## Hands-on Exercise

### Your Task

Practice the complete Git workflow:

1. **Initialize**: Create a new folder and init Git
2. **First commit**: Create index.html, commit
3. **Branch**: Create `feature-nav` branch
4. **Develop**: Add navigation, commit
5. **Switch**: Go back to main
6. **Merge**: Merge feature-nav into main
7. **Remote**: Create GitHub repo, push
8. **Collaborate**: Make a change on GitHub, pull locally

### Challenge

Create a conflict intentionally and resolve it.

---

## Summary

✅ `git init` / `git clone` to start
✅ `git add` + `git commit` to save changes
✅ **Branches** isolate feature development
✅ `git merge` combines branches
✅ **Remotes** enable collaboration
✅ **Pull requests** for code review
✅ **.gitignore** prevents tracking sensitive/generated files
✅ **Conventional commits** keep history readable
✅ **Conflict resolution** when same lines are modified

**Back to:** [Development Tools Overview](./00-development-tools.md)

---

## Further Reading

- [Pro Git Book](https://git-scm.com/book/en/v2) - Free, comprehensive
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet/)

<!-- 
Sources Consulted:
- Git Documentation: https://git-scm.com/doc
- GitHub Docs: https://docs.github.com
-->
