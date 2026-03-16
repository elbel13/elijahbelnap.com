# Copilot Instructions

Personal website and blog for Elijah Belnap, built with Jekyll and deployed to GitHub Pages.

## Build & Serve

```bash
bundle install
bundle exec jekyll serve   # http://localhost:4000
bundle exec jekyll build   # outputs to _site/
```

## Key Conventions

### SASS Directory
The SASS source lives in **`_ssas/`** (not the standard `_sass/`). The single file `_ssas/minima.scss` defines theme variables and imports Minima's partials. Compiled output is `assets/main.css`.

### Front Matter
**Posts** (`_posts/YYYY-MM-DD-title.md`):
```yaml
---
layout: post
title: "Post Title"
author: Elijah Belnap
tags: tag1 tag2
category: category-slug
---
```

Each post's `category` creates a subdirectory in the built site (e.g., `category: sql` → `/sql/post-title/`).

**Pages** (`*.markdown` at root):
```yaml
---
layout: default   # or "home" for index.markdown
title: Page Title
---
```

### Images
Post images go in `assets/images/` and are referenced in markdown:
```markdown
![Alt text](/assets/images/image-name.png)
```
