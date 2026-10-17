---
name: use-website-repo
description: Use if the user asks you for a website (or web page) they can see, e.g. on their phone. Builds the page in the public claude-chat-website repo at ~/claude/claude-chat-website and publishes it with GitHub Pages.
---

# Publish a web page to claude-chat-website

The user wants a page they can open in a browser, usually on their phone. Build it in `~/claude/claude-chat-website`, which GitHub Pages publishes at <https://egodfrey-sunflower.github.io/claude-chat-website/>.

Before writing anything, read `~/claude/claude-chat-website/CLAUDE.md`. It covers the layout, the page guidelines and the privacy rules.

## ⚠️ The repo is public

Never commit personal details about the user: names, contact details, addresses, travel plans or where they're staying, health, money, work or family details, or secrets of any kind. Write the page for a general reader. If the page needs personal details to be useful, don't publish it here. Tell the user, and offer another route such as a private artifact or sending the file directly. Before committing, re-read the diff for anything personal.

## Steps

1. Get today's path with `date +%Y%m/%d` and create the directory if needed.
2. Write the page as `YYYYMM/DD/<page-name>.html`. Put any assets in `YYYYMM/DD/<page-name>/`.
3. Add a link to the page in the root `index.html`, newest first.
4. Check the diff for personal details, then commit with a message like `page: <page-name>` and push to `origin main`. Pushing is part of the job here: the page isn't visible until it's pushed.
5. Give the user the live URL: `https://egodfrey-sunflower.github.io/claude-chat-website/YYYYMM/DD/<page-name>.html`. Pages usually takes a minute or two to deploy after a push.

To change an existing page, edit it in place, commit as `page: <page-name> (update)`, and push.
