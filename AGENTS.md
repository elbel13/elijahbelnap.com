# AGENTS.md - Agentic Coding Guidelines for elijahbelnap.com

This is the source code for Elijah Belnap's personal blog, built with Jekyll and deployed to GitHub Pages.

## Build & Development Commands

### Installation
```bash
bundle install
```

### Local Development
```bash
bundle exec jekyll serve    # http://localhost:4000
bundle exec jekyll build    # outputs to ./_site/
```

### CI/CD
Auto-deploys to GitHub Pages via `.github/workflows/jekyll.yml` on pushes to `main` or `fix-images` branches.

---

## Code Style Guidelines

### General Principles
- Static Jekyll site using Liquid templates and Markdown
- Keep files minimal - only include what's necessary
- Follow existing conventions in the codebase

### Markdown Formatting
- Use standard GitHub Flavored Markdown (GFM)
- Code blocks specify language for syntax highlighting (e.g., ```sql)
- Use ATX-style headers (`#`, `##`, `###`)
- Wrap inline code with backticks

### Liquid Templates
- Use `{{ "{% " }} %}` for logic tags
- Use `{{ "{{ " }}` for output tags
- Use `| relative_url` filter for internal links and images

### YAML Front Matter

#### Posts (`_posts/YYYY-MM-DD-title.md`)
```yaml
---
layout: post
title: "Post Title"
author: Elijah Belnap
tags: tag1 tag2
category: category-slug
---
```
- Date: `YYYY-MM-DD`, Title case, Tags: lowercase

#### Pages (`*.markdown` at root)
```yaml
---
layout: default
title: Page Title
---
```

### SCSS/SASS
- Source in `_ssas/` (custom, not standard `_sass/`)
- Entry: `_ssas/minima.scss` defines variables and imports Minima
- Output: `assets/main.css`
- Use `!default` for variable overrides

### File Paths
- Markdown images: `/assets/images/image-name.png`
- Templates: `{{ "/assets/images/image.png" | relative_url }}`

---

## File Organization

```
elijahbelnap.com/
├── _posts/           # Blog posts (YYYY-MM-DD-title.md)
├── _layouts/        # Custom layouts
├── _includes/       # Reusable HTML partials
├── _ssas/           # SCSS source
├── _data/           # YAML data files
├── _config.yml      # Jekyll configuration
├── assets/images/   # All images
├── *.markdown       # Pages (index, blog, about)
└── .github/workflows/# CI/CD
```

---

## Content Conventions

### Writing Blog Posts
1. Create `_posts/YYYY-MM-DD-title.md` with front matter
2. Add hero image to `assets/images/` and reference in post
3. Category determines URL (`category: sql` → `/sql/post-title/`)
4. Use descriptive, specific tags

### Images
- Place all in `assets/images/`
- Markdown: `![Alt](/assets/images/image.png)`
- Templates: `{{ "/assets/images/image.png" | relative_url }}`

### Code Examples
- Use fenced code blocks with language identifier
- Keep code concise and focused

---

## Testing & Validation

No traditional tests. Validate by building:

```bash
bundle exec jekyll build
bundle exec jekyll serve  # manual testing
```

### Pre-commit Checklist
- [ ] Run `bundle exec jekyll build` - no errors
- [ ] Verify images load correctly
- [ ] Check internal links work
- [ ] Validate YAML syntax

### Common Issues
- **Broken links**: Use `| relative_url` filter
- **Missing images**: Check `assets/images/`
- **Build errors**: Validate YAML 2-space indentation

---

## Dependencies

- Jekyll ~> 4.3.3
- minima ~> 2.5 (theme)
- jekyll-feed, jekyll-sitemap, jekyll-seo-tag

---

## Related Documentation
- [Jekyll Documentation](https://jekyllrb.com/)
- [Minima Theme](https://github.com/jekyll/minima)
- [GitHub Pages](https://docs.github.com/en/pages)
